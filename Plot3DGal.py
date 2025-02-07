#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:05:24 2024

@author: alexey
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import pyvista as pv
import os
import legwork.source as source
import legwork.visualisation as vis
import astropy.units as u
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as mcm  # Import matplotlib's cm module as mcm to avoid confusion
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import cmasher as cmr

sys.path.insert(1, './PyModules/')


Make3DMap      = True
MakeGalView    = True
DWDAnalyse     = True
LEGWORKStepQ   = False
AnimationStepQ = False


#SimType can be MS, DWDCand, DWDLISA
SimTypesUse   = ['MS', 'DWDCand', 'DWDLISA']
#Plot group name
PlotSetName   = '20241107ICs'

#RunWave         = 'fiducial'
#RunWave         = 'porb_log_uniform'
RunWave         = 'uniform_ecc'
#RunWave         = 'qmin_01'
#Code            = 'COSMIC'
#Code            = 'ComBinE'
Code            = 'COMPAS'
    
OutputSubfolder = 'ICVariations'
CurrOutDir      = './ProcessedSimulations/' + OutputSubfolder + '/' + RunWave 
PathSet         = {'MS': './FullGalaxy.csv', 'DWDCand': CurrOutDir + '/FullGalaxyDWD.csv', 'DWDLISA': CurrOutDir + '/FullGalaxyDWDLISAOnly.csv'}
GalStatsPath    = CurrOutDir + '/GalaxyLISACandidateStats.csv'

FileStringSet = {'MS': 'MS', 'DWDCand': 'DWD', 'DWDLISA': 'DWDLISA'}
SubfolderSet  = {'MS': 'MS', 'DWDCand': 'DWD', 'DWDLISA': 'DWDLISA'}
LongFigPaths  = ['./Figures/' + PlotSetName + '/' + RunWave + '/' + Code + '/' + SubfolderSet[SimTypesUse[iType]] for iType in range(len(SimTypesUse))]

# Read in the Galaxy data
dfSet       = [pd.read_csv(PathSet[SimTypesUse[iType]]) for iType in range(len(SimTypesUse))]
dfPointers  = {SimTypesUse[iType]:iType  for iType in range(len(SimTypesUse))}

#Stats printout
if ('DWDCand' in SimTypesUse) or ('DWDLISA' in SimTypesUse):
    statsDF = pd.read_csv(GalStatsPath)
    print('Galactic number of present-day LISA candidates: ', int(np.sum(statsDF['SubBinNDWDsReal'])))
    DWDCands = int(np.sum(statsDF['SubBinNDWDsReal']))
    if ('DWDLISA' in SimTypesUse):
        print('Galactic number of present-day LISA-observable DWDs: ',  int(np.sum(statsDF['SubBinNDWDsReal'])*(len(dfSet[dfPointers['DWDLISA']].index))/(len(dfSet[dfPointers['DWDCand']].index))))
        DWDLISA =  int(np.sum(statsDF['SubBinNDWDsReal'])*(len(dfSet[dfPointers['DWDLISA']].index))/(len(dfSet[dfPointers['DWDCand']].index)))


#For similar numbers in plotting
if ('DWDCand' in SimTypesUse):
    dfSet[dfPointers['DWDCand']] = dfSet[dfPointers['DWDCand']].sample(n=10000, replace=True)


# Extracting the columns
xSet = []
ySet = []
zSet = []
for i in range(len(SimTypesUse)):
    df     = dfSet[i]
    xSet.append(df['Xkpc'])
    ySet.append(df['Ykpc'])
    zSet.append(df['Zkpc'])
    
    os.makedirs(LongFigPaths[i],exist_ok=True)


if Make3DMap:
    for iType in SimTypesUse:
        x = xSet[dfPointers[iType]]
        y = ySet[dfPointers[iType]]
        z = zSet[dfPointers[iType]]
        
        # Create a new figure for the 3D plot with high DPI
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
        # Plotting the points
        ax.scatter(x, y, z, color='lime', s=5, alpha=0.05)  # 's' is the size of the points
    
        # Set the background color
        ax.set_facecolor('black')
    
        AxRange = 15
    
        # Set the range for all axes
        ax.set_xlim([-AxRange, AxRange])
        ax.set_ylim([-AxRange, AxRange])
        ax.set_zlim([-AxRange, AxRange])
    
        # Set the labels and title with colors
        ax.set_xlabel('X, kpc', color='white')
        ax.set_ylabel('Y, kpc', color='white')
        ax.set_zlabel('Z, kpc', color='white')
    
        # Change the color of the tick labels to white
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
    
        # Save the plot as a high-resolution image
        plt.savefig(LongFigPaths[dfPointers[iType]] + '/LISAGal' + FileStringSet[iType] + '.png', dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='black')

if MakeGalView:
    for iType in SimTypesUse:
        
        df = dfSet[dfPointers[iType]]
        
        #Pre-setup plot options, suited for an 8cm MNRAS single-column figure
        cm  = 1/2.54  # centimeters in inches
        DPI = 300
        LW  = 3       
        plt.clf()
        #plt.rc('font', family='fantasy', size=11)
        #plt.rc('font', family='')
        #plt.rcParams["font.family"] = "fantasy"
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['mathtext.fontset'] = 'cm'
        sns.set_style("white")
        sns.set_context("paper")
    
    
        fig = plt.figure(figsize=(14*cm,7*cm), constrained_layout=True)
        
        ax = plt.gca()    
        scatter = ax.scatter(df["Gall"]*(180/(np.pi)),df["Galb"]*(180/(np.pi)),c=df['Age'],s=1, cmap='cmr.cosmic')
               
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        plt.xlim([0, 360])  
        plt.ylim([-90, 90])  
    
        plt.xlabel(r'l, deg')
        plt.ylabel(r'b, deg')
        plt.text(30, 60, 'DWD Candidates: ' + str(DWDCands) + '\n' + 'LISA DWDs: ' + str(DWDLISA))
        #plt.legend(loc="right", fontsize=7)
        #fig.colorbar(cm.ScalarMappable(cmap='cmr.cosmic'), ax=ax)
        #fig.colorbar(mcm.ScalarMappable(cmap='cmr.cosmic'), ax=ax)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Age (Gyr)', rotation=270, labelpad=10)
        
        plt.savefig(LongFigPaths[dfPointers[iType]] + '/GalaxyCoordsAge' + FileStringSet[iType] + '.png', dpi=DPI)   
    
        df['FeH'][df['FeH'] >0.4] = 0.4
    
        fig = plt.figure(figsize=(14*cm,7*cm), constrained_layout=True)
        
        ax = plt.gca()    
        scatter = ax.scatter(df["Gall"]*(180/(np.pi)),df["Galb"]*(180/(np.pi)),c=df['FeH'],s=1, cmap='cmr.cosmic')
                       
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        plt.xlim([0, 360])  
        plt.ylim([-90, 90])  
    
        
        plt.xlabel(r'l, deg')
        plt.ylabel(r'b, deg')
        #plt.legend(loc="right", fontsize=7)
        #fig.colorbar(cm.ScalarMappable(cmap='cmr.cosmic'), ax=ax)
        #fig.colorbar(mcm.ScalarMappable(cmap='cmr.cosmic'), ax=ax)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('[Fe/H]', rotation=270, labelpad=10)
        
        plt.savefig(LongFigPaths[dfPointers[iType]] + '/GalaxyCoordsFeH' + FileStringSet[iType] + '.png', dpi=DPI) 
        
        
        fig = plt.figure(figsize=(12*cm,12*cm), constrained_layout=True)
        
        ax = plt.gca()    
        ax.set_aspect('equal')
        scatter = ax.scatter(df["Xkpc"],df["Ykpc"],c=df['Age'],s=1, cmap='cmr.cosmic')
               
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        plt.xlim([-10, 10])  
        plt.ylim([-10, 10])  
    
        plt.xlabel(r'x, kpc')
        plt.ylabel(r'y, kpc')
        #plt.legend(loc="right", fontsize=7)
        #fig.colorbar(cm.ScalarMappable(cmap='cmr.cosmic'), ax=ax)
        #fig.colorbar(mcm.ScalarMappable(cmap='cmr.cosmic'), ax=ax)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Age (Gyr)', rotation=270, labelpad=10)
        
        plt.savefig(LongFigPaths[dfPointers[iType]] + '/CartesianXYAge' + FileStringSet[iType] + '.png', dpi=DPI, bbox_inches='tight')   
    
        df['FeH'][df['FeH'] >0.4] = 0.4
    
        fig = plt.figure(figsize=(14*cm,7*cm), constrained_layout=True)
        
        ax = plt.gca()    
        scatter = ax.scatter(df["Gall"]*(180/(np.pi)),df["Galb"]*(180/(np.pi)),c=df['FeH'],s=1, cmap='cmr.cosmic')
                       
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        plt.xlim([0, 360])  
        plt.ylim([-90, 90])  
    
        
        plt.xlabel(r'l, deg')
        plt.ylabel(r'b, deg')
        #plt.legend(loc="right", fontsize=7)
        #fig.colorbar(cm.ScalarMappable(cmap='cmr.cosmic'), ax=ax)
        #fig.colorbar(mcm.ScalarMappable(cmap='cmr.cosmic'), ax=ax)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('[Fe/H]', rotation=270, labelpad=10)
        
        plt.savefig(LongFigPaths[dfPointers[iType]] + '/GalaxyCoordsFeH' + FileStringSet[iType] + '.png', dpi=DPI) 
        
        
        
        cm_to_inch = 1 / 2.54
        nrows=4
        ncols=3
        bin_values = sorted(df['Bin'].unique())
        n_bins = len(bin_values)
        
        fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14 * cm_to_inch * ncols, 7 * cm_to_inch * nrows),
        constrained_layout=True)
    
        axes = axes.flatten()
        
        vmin = df['FeH'].min()
        vmax = df['FeH'].max()
        
        Labels = np.array(['Thin Disk 1', 'Thin Disk 2','Thin Disk 3','Thin Disk 4','Thin Disk 5','Thin Disk 6','Thin Disk 7','Thick Disk','Halo','Bulge'])
        for i, (bin_value, ax) in enumerate(zip(bin_values, axes)):
            # Filter DataFrame for the current 'Bin' value
            df_bin = df[df['Bin'] == bin_value]
        
            # Create scatter plot
            scatter = ax.scatter(
                df_bin["Gall"] * (180 / np.pi),
                df_bin["Galb"] * (180 / np.pi),
                c=df_bin['FeH'],
                s=1,
                cmap='cmr.cosmic',
                vmin=vmin,
                vmax=vmax
            )
    
            # Set axis limits and labels
            ax.set_xlim([0, 360])
            ax.set_ylim([-90, 90])
            ax.set_xlabel('l, deg')
            ax.set_ylabel('b, deg')
            ax.set_title(Labels[i])
                       
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        plt.xlim([0, 360])  
        plt.ylim([-90, 90])  
    
        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
    
        # Create a ScalarMappable for the colorbar
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cmr.cosmic
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for ScalarMappable
    
        # Add the colorbar to the figure
        cbar = fig.colorbar(
            sm,
            ax=axes.ravel().tolist(),
            orientation='vertical',
            fraction=0.02,
            pad=0.04
        )
        cbar.set_label('[Fe/H]', rotation=270, labelpad=15)
        
        plt.savefig(LongFigPaths[dfPointers[iType]] + '/GalaxyCoordsFeHBinned' + FileStringSet[iType] + '.png', dpi=DPI) 
        
if DWDAnalyse:
    DWDAll   = dfSet[dfPointers['DWDCand']]
    DWDLISA  = dfSet[dfPointers['DWDLISA']]

    # Pre-setup plot options, suited for an 8cm MNRAS single-column figure
    cm  = 1/2.54  # centimeters in inches
    DPI = 300
    LW  = 3       
    plt.clf()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['mathtext.fontset'] = 'cm'
    
    # Adjust font sizes to fit within the small figure
    sns.set_style("white")
    sns.set_context("paper", rc={
        "axes.labelsize": 8,   # x and y labels
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "font.size": 8         # general font size
    })
    
    def plot_dataframe_P(df, filename):
        # Calculate the total mass
        df['M_tot'] = df['mass1'] + df['mass2']
    
        # Set up figure size
        width_cm = 8  # 8 cm width for MNRAS single-column figure
        aspect_ratio = 0.8  # width / height
        width = width_cm * cm  # Convert width to inches
        height = width * aspect_ratio  # Calculate height to maintain aspect ratio
    
        fig, ax = plt.subplots(figsize=(width, height), dpi=DPI)
    
        # Choose a vivid colormap
        cmap = cmr.cosmic  # Alternatively, try cmr.guppy or cmr.flamingo
    
        # Create scatter plot
        scatter = ax.scatter(
            df['SubBinMidAge'],
            df['PSetTodayHours'],
            c=df['M_tot'],
            cmap=cmap,
            s=1,
            vmin=0.5,
            vmax=2.0,
            edgecolors='none'
        )
    
        # Set axis labels and limits
        ax.set_xlabel('Age, Myr', fontsize=8)
        ax.set_ylabel('P, hours', fontsize=8)
        ax.set_xlim(0, 11000)
        ax.set_ylim(0, 1.5)
    
        # Set axes aspect ratio to 'auto' to prevent squeezing
        ax.set_aspect('auto')
    
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'M$_{\rm tot},\,{\rm M}_\odot$', fontsize=8)
    
        # Adjust layout to ensure labels are not cut off
        plt.tight_layout()
    
        # Save figure with bbox_inches='tight' to include all elements
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()

    def plot_dataframe_q(df, filename):
        # Calculate the total mass
        df['M_tot'] = df['mass1'] + df['mass2']
    
        # Set up figure size
        width_cm = 8  # 8 cm width for MNRAS single-column figure
        aspect_ratio = 0.8  # width / height
        width = width_cm * cm  # Convert width to inches
        height = width * aspect_ratio  # Calculate height to maintain aspect ratio
    
        fig, ax = plt.subplots(figsize=(width, height), dpi=DPI)
    
        # Choose a vivid colormap
        cmap = cmr.cosmic  # Alternatively, try cmr.guppy or cmr.flamingo
    
        # Create scatter plot
        scatter = ax.scatter(
            df['SubBinMidAge'],
            df['qSet'],
            c=df['M_tot'],
            cmap=cmap,
            s=1,
            vmin=0.5,
            vmax=2.0,
            edgecolors='none'
        )
    
        # Set axis labels and limits
        ax.set_xlabel('Age, Myr', fontsize=8)
        ax.set_ylabel(r'q$=(M_{\rm min}/M_{\rm max})$', fontsize=8)
        ax.set_xlim(0, 11000)
        ax.set_ylim(0, 1)
    
        # Set axes aspect ratio to 'auto' to prevent squeezing
        ax.set_aspect('auto')
    
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'M$_{\rm tot},\,{\rm M}_\odot$', fontsize=8)
    
        # Adjust layout to ensure labels are not cut off
        plt.tight_layout()
    
        # Save figure with bbox_inches='tight' to include all elements
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        plt.close()

    # Generate plots for DWDAll and DWDLISA DataFrames
    plot_dataframe_q(DWDAll, LongFigPaths[dfPointers['DWDCand']] + '/qvsAge' + FileStringSet['DWDCand'] + '.png')
    plot_dataframe_q(DWDLISA, LongFigPaths[dfPointers['DWDLISA']] + '/qvsAge' + FileStringSet['DWDLISA'] + '.png')        
    plot_dataframe_P(DWDAll, LongFigPaths[dfPointers['DWDCand']] + '/PvsAge' + FileStringSet['DWDCand'] + '.png')
    plot_dataframe_P(DWDLISA, LongFigPaths[dfPointers['DWDLISA']] + '/PvsAge' + FileStringSet['DWDLISA'] + '.png')     


#LEGWORK part
if LEGWORKStepQ:

    n_values = len(df.index)

    m_1    = (df['mass1']).to_numpy() * u.Msun
    m_2    = (df['mass2']).to_numpy() * u.Msun
    dist   = (df['RRelkpc']).to_numpy() * u.kpc
    f_orb = (1/(df['PSetTodayHours']*60*60)).to_numpy() * u.Hz
    ecc   = np.zeros(n_values)
    
    sources = source.Source(m_1=m_1, m_2=m_2, ecc=ecc, dist=dist, f_orb=f_orb)
    snr     = sources.get_snr(verbose=True)
    
    cutoff = 7
    
    detectable_threshold = cutoff
    detectable_sources   = sources.snr > cutoff
    print("{} of the {} sources are detectable".format(len(sources.snr[detectable_sources]), n_values))
    
    # create the same plot but set `show=False`
    fig, ax = sources.plot_source_variables(xstr="f_orb", ystr="snr", disttype="kde", log_scale=(True, True),
                                            fill=True, show=False, which_sources=sources.snr > 7, figsize=(10, 8))
    
    #ax.set_aspect(0.8)
    
    # duplicate the x axis and plot the LISA sensitivity curve
    right_ax = ax.twinx()
    frequency_range = np.logspace(np.log10(2e-6), np.log10(2e-1), 1000) * u.Hz
    vis.plot_sensitivity_curve(frequency_range=frequency_range, fig=fig, ax=right_ax,fill=False)
    
    # Adjust layout to prevent cropping
    fig.tight_layout()
    
    fig.savefig('./SensitivityCurveLISAOnly.png', bbox_inches='tight')

if AnimationStepQ:
    
    # Ensure the output directory exists
    output_dir = './Frames/'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the galaxy data
    data = df
    
    # Extract X, Y, Z coordinates
    points = data[['Xkpc', 'Ykpc', 'Zkpc']].values
    
    # Create a PyVista point cloud
    point_cloud = pv.PolyData(points)
    
    # Initialize the plotter
    plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
    plotter.set_background('black')
    
    # Add the point cloud to the plotter with styling
    plotter.add_mesh(
        point_cloud,
        color='white',
        point_size=0.5,
        render_points_as_spheres=True,
        opacity=0.8,
        )
    
    # Camera and animation settings
    num_frames = 360  # Total number of frames
    fps = 30  # Frames per second
    radius = np.max(np.linalg.norm(points[:, :2], axis=1)) * 1.2  # Orbit radius
    elevation_angle = np.radians(30)  # Elevation angle above the galactic plane
    angles = np.linspace(0, 4 * np.pi, num_frames)  # Two full orbits
    
    # Generate and save each frame
    for i, angle in enumerate(angles):
        # Calculate camera position
        x = radius * np.cos(angle) * np.cos(elevation_angle)
        y = radius * np.sin(angle) * np.cos(elevation_angle)
        z = radius * np.sin(elevation_angle)
        
        # Update camera settings
        plotter.camera_position = [
            (x, y, z),  # Camera position
            (0, 0, 0),  # Look at the galactic center
            (0, 0, 1),  # Up direction
            ]
        plotter.camera.view_angle = 60  # Human eye field of view
        
        
        # Render and save the frame
        filename = os.path.join(output_dir, f'frame_{i:04d}.png')
        plotter.show(screenshot=filename, auto_close=False)
        plotter.clear()  # Clear the plotter for the next frame
        sys.exit()
    # Close the plotter after rendering
    plotter.close()
