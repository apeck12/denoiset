import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time


class TomogramCurator:
    
    def __init__(self, in_file: str):
        """
        Set up tomogram monitoring and curation class.
        -- Thickness: in Angstrom, select thicker than threshold
        -- Tilt_Axis: in degrees, select within threshold degrees of median
        -- Global_Shift: in Angstrom, select less than threshold
        -- Bad_Patch_Low, Bad_Patch_All: fraction, select less than threshold
        -- CTF_Res: in Angstrom, select less than threshold
        -- CTF_Score: fraction, select greater than threshold
                
        Parameters
        ----------
        in_file: path to AreTomo3's TiltSeries_Metrics.csv 
        """
        self.in_file = in_file
        self.reload()
        self.criteria = {'Thickness': 1000,
                         'Tilt_Axis': 1.5,
                         'Global_Shift': 750,
                         'Bad_Patch_Low': 0.05,
                         'Bad_Patch_All': 0.1,
                         'CTF_Res': 20.0,
                         'CTF_Score': 0.05}
        self.indices = range(len(self.df))
        self.dataset_size = 0
        
    def reload(self):
        """
        Reload the continuously-updated metrics csv file. Thickness 
        and global shifts are converted from pixels to Angstrom and
        the (A) suffix is dropped from the relevant column names.
        """
        df = pd.read_csv(self.in_file)
        df['Thickness(Pix)'] = df['Thickness(Pix)'].values * df['Pix_Size(A)'].values
        df['Global_Shift(Pix)'] = df['Global_Shift(Pix)'].values * df['Pix_Size(A)'].values
        df.rename(columns={'Thickness(Pix)': 'Thickness', 
                           'Global_Shift(Pix)': 'Global_Shift', 
                           'CTF_Res(A)': 'CTF_Res'}, 
                  inplace=True)
        self.df = df
        
    def reset_criterion(self, metric: str, threshold: float):
        """
        Reset the threshold for the specified metric.
        
        Parameters
        ----------
        metric: metric name as in AreTomo3 CSV header
        threshold: updated threshold value
        """
        if metric in self.criteria.keys():
            self.criteria[metric] = threshold
        else:
            raise ValueError("Requested metric not found.")
            
    def sort_selected(self, max_selected: int, sort_by: str):
        """
        Sort the selected tomograms by the given metric and only
        retain the top max_selected tomograms.
        
        Parameters
        ----------
        max_selected: maximum number of tomograms to keep
        sort_by: metric by which to sort the retained tomograms        
        """
        if sort_by in ['Global_Shift', 'Bad_Patch_Low', 'Bad_Patch_All', 'CTF_Res']:
            tc_sel = self.df.iloc[self.indices].sort_values(by=[sort_by], ascending=True)
        elif sort_by in ['CTF_Score', 'Thickness']:
            tc_sel = self.df.iloc[self.indices].sort_values(by=[sort_by], ascending=False)
        elif sort_by == 'Tilt_Axis':
            tilt_axis_ref = np.median(self.df.iloc[self.indices].Tilt_Axis.values)
            sort_indices = np.argsort(np.abs(self.df.iloc[self.indices].Tilt_Axis.values - tilt_axis_ref))
            tc_sel = self.df.iloc[self.indices].iloc[sort_indices]
        else:
            raise ValueError("Unrecognized sort_by argument")
        self.indices = tc_sel.index[:max_selected]
                
    def curate(self, out_file: str=None, vol_path: str=None, max_selected: int=50, sort_by: str='Global_Shift'):
        """
        Get a curated list of tomograms that meet criteria. If the 
        selected set is larger than max_retained, the set is sorted
        by the specified metric and only the max_retained tomograms
        according to that metric are kept.
        
        Parameters
        ----------
        out_file: output listing curated files prefixes
        vol_path: directory containing volumes
        max_selected: maximum number of tomograms to keep
        sort_by: metric by which to sort the retained tomograms
        """
        conditions = {}
        for i,metric in enumerate(self.criteria.keys()):
            if metric == 'Tilt_Axis':
                conditions[i] = np.abs(self.df[metric].values - np.median(self.df[metric].values)) < self.criteria[metric]
            elif metric == 'CTF_Score' or metric == 'Thickness':
                conditions[i] = self.df[metric].values > self.criteria[metric]
            else:
                conditions[i] = self.df[metric].values <= self.criteria[metric]
            print(f"{metric}, threshold: {self.criteria[metric]}, {np.sum(conditions[i])}/{len(self.df)} tilt-series retained")

        conditions_met = np.array([conditions[i] for i in conditions])
        self.indices = np.where(np.sum(conditions_met, axis=0)==len(conditions.keys()))[0]
        print(f"Combined thresholds: {len(self.indices)}/{len(self.df)} tilt-series retained")
        
        if len(self.indices) == 0:
            print("Warning: no tomograms were retained based on current thresholds.")
            return

        if len(self.indices) > max_selected:
            self.sort_selected(max_selected, sort_by)
            print(f"Retaining top {max_selected} tomograms based on {sort_by}")
            
        basenames = self.df.iloc[self.indices].Tilt_Series.values
        basenames = [os.path.splitext(fn)[0] for fn in basenames]
        if vol_path is None:
            vol_path = os.path.dirname(self.in_file)
        full_paths = np.array([os.path.join(vol_path, fn) for fn in basenames])
        
        if out_file is not None:
            np.savetxt(out_file, full_paths, fmt="%s")
        return full_paths
        
    def visualize_curated(self, nbins: int=50, out_file: str=None):
        """
        Visualize statistics for the selected set versus all.
        
        Parameters
        ----------
        nbins: number of histogram bins
        out_file: output path for png
        """
        f, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(12,6))

        ax_list = [ax1,ax2,ax3,ax4,ax5,ax6]
        metrics = ['Thickness', 'Tilt_Axis', 'Global_Shift', 'Bad_Patch_Low', 'CTF_Res', 'CTF_Score']
        xlabels = ['Thickness (Å)', 'Tilt Axis ($^{\circ}$)', 'Global Shift (Å)',
                  'Fraction of bad patches, low tilts', 'CTF resolution (Å)', 'CTF Score (Å)']

        for i,ax in enumerate(ax_list):
            bmin, bmax = self.df[metrics[i]].values.min(), self.df[metrics[i]].values.max()
            bins = np.linspace(bmin,bmax,nbins)
            ax.hist(self.df[metrics[i]].values, color='dimgrey', alpha=0.5, bins=bins, label="All tilt-series")
            ax.hist(self.df.iloc[self.indices][metrics[i]].values, color='darkred', alpha=0.5, bins=bins, label="Selected")
            ax.set_xlabel(xlabels[i], fontsize=12)

        for ax in [ax1,ax4]:
            ax.set_ylabel("Number of tomograms", fontsize=12)
        ax3.legend()    

        f.subplots_adjust(hspace=0.3)
        plt.show()
        
        if out_file is not None:
            f.savefig(out_file, bbox_inches="tight", dpi=300)
            
    def select_dataframe(self, out_file: str=None)->pd.DataFrame:
        """
        Save the selected dataframe.
        
        Parameters
        ----------
        out_file: output path for CSV file
        """
        df_sel = self.df.iloc[self.indices]
        if out_file is not None:
            df_sel.to_csv(out_file, index=False)
        return df_sel
    
    def curate_live(
        self, out_file: str, vol_path: str, max_selected: int=50, min_selected: int=20, 
        sort_by: str='Global_Shift', t_interval: int=300, t_exit: int=1800,
    ):
        """
        Select from a continuously-updated TiltSeries_Metrics.csv file.
        
        Parameters
        ----------
        out_file: output listing curated files prefixes
        vol_path: directory containing volumes
        max_selected: maximum number of tomograms to keep
        min_selected: minimum number of tomograms to keep
        sort_by: metric by which to sort the retained tomograms
        t_interval: interval between checking for new tomograms in seconds
        t_exit: interval after which to exit in seconds if no new tomograms are found
        """
        start_time = time.time()
        while True:
            self.reload()
            self.curate(
                out_file = out_file, 
                vol_path = vol_path,
                max_selected = max_selected,
                sort_by = sort_by,
            )
            self.visualize_curated(
                out_file = os.path.join(os.path.dirname(out_file), "traininglist.png"),
            )
            
            if len(self.indices) >= min_selected:
                break
            
            if len(self.df) > self.dataset_size:
                self.dataset_size = len(self.df)
                start_time = time.time()
            else:
                print("No new tomograms found")
                
            time.sleep(t_interval)
            t_elapsed = time.time() - start_time
            if t_elapsed > t_exit:
                print(f"Only {len(self.indices)} high-quality tomograms detected")
                if len(self.indices) < 1:
                    raise ValueError("No high-quality tomograms were found")
                break
