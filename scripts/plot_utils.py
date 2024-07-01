import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


line_style = {
    'top_fine_tune':'dotted',
    'top':'-',
    'tau_fine_tune':'dotted',
    'tau':'-',
    'eic_fine_tune':'dotted',
    'eic':'-',

    'qg_fine_tune':'dotted',
    'qg':'-',
    'cms_fine_tune':'dotted',
    'cms':'-',

    'jetnet30_fine_tune':'dotted',
    'jetnet30':'-',
    'jetnet150_fine_tune':'dotted',
    'jetnet150':'-',
    'omnifold_fine_tune':'dotted',
    'omnifold':'-',
    'lhco_fine_tune':'dotted',
    'lhco':'-',
    'jetclass':'-',
    'h1_fine_tune':'dotted',
    'h1':'-',
    'atlas_fine_tune':'dotted',
    'atlas':'-',

    'atlas_small_fine_tune':'dotted',
    'atlas_small':'-',

    't_truth':'-',
    't_gen':'dotted',
    'q_truth':'-',
    'q_gen':'dotted',
    'g_truth':'-',
    'g_gen':'dotted',
    'z_truth':'-',
    'z_gen':'dotted',
    'w_truth':'-',
    'w_gen':'dotted',

    'eic_truth':'-',
    'eic_gen':'dotted',
    
    'true':'-',
    'gen':'dotted',

    'pythia':'dotted',
    'pythia_unfolded_baseline':'dotted',
    'pythia_unfolded_fine_tune':'dotted',
    'pythia_reweighted_baseline':'dotted',
    'pythia_reweighted_fine_tune':'dotted',
    
    'herwig':'-',
    
    
}

colors = {
    'top_fine_tune':'#7570b3',
    'top':'#7570b3',
    'tau_fine_tune':'#7570b3',
    'tau':'#7570b3',
    'eic_fine_tune':'#7570b3',
    'eic':'#7570b3',
    'qg_fine_tune':'#d95f02',
    'qg':'#d95f02',
    'jetnet30_fine_tune':'#1b9e77',
    'jetnet30':'#1b9e77',
    'jetnet150_fine_tune':'#e7298a',
    'jetnet150':'#e7298a',
    'omnifold_fine_tune':'darkorange',
    'omnifold':'darkorange',
    'lhco_fine_tune':'#917E9C',
    'lhco':'#fc8d59',
    'jetclass':'black',

    't_truth':'#1b9e77',
    't_gen':'#1b9e77',
    'q_truth':'#e7298a',
    'q_gen':'#e7298a',
    'g_truth':'#1b9e77',
    'g_gen':'#1b9e77',
    'w_truth':'#1b9e77',
    'w_gen':'#1b9e77',
    'z_truth':'#1b9e77',
    'z_gen':'#1b9e77',
    
    'h1':'#d95f02',
    'h1_fine_tune':'#d95f02',
    'atlas_fine_tune':'#7570b3',
    'atlas':'#7570b3',
    'cms_fine_tune':'#7570b3',
    'cms':'#7570b3',

    'atlas_small_fine_tune':'#7570b3',
    'atlas_small':'#7570b3',
    'true':'black',
    'gen':'#1b9e77',
    
    'eic_truth':'#1b9e77',
    'eic_gen':'#1b9e77',
    
    'herwig': 'black',
    'pythia':'darkorange',
    'pythia_unfolded_baseline':'red',
    'pythia_unfolded_fine_tune':'#7570b3',
    'pythia_reweighted_baseline':'red',
    'pythia_reweighted_fine_tune':'#7570b3',
}

name_translate = {
    'top_fine_tune':'Top tagging OmniLearn',
    'top':'Top tagging',
    'tau_fine_tune':'Tau tagging OmniLearn',
    'tau':'Tau tagging',

    'eic_fine_tune':'EIC Pythia OmniLearn',
    'eic':'EIC Pythia',
    'qg_fine_tune':'Quark/Gluon OmniLearn',
    'qg':'Quark/Gluon',
    'jetnet30_fine_tune':'JetNet30 OmniLearn',
    'jetnet30':'JetNet30',
    'jetnet150_fine_tune':'JetNet150 OmniLearn',
    'jetnet150':'JetNet150',
    'omnifold_fine_tune':'OmniFold Z+Jets OmniLearn',
    'omnifold':'OmniFold Z+Jets',
    'lhco_fine_tune':'LHCO R&D OmniLearn',
    'lhco':'LHCO R&D',
    'h1':'DIS H1',
    'h1_fine_tune':'DIS H1 OmniLearn',
    'atlas_fine_tune':'ATLAS Top tagging (40M) OmniLearn',
    'atlas':'ATLAS Top tagging (40M)',
    'cms_fine_tune':'CMS Jets OmniLearn',
    'cms':'CMS Jets',

    'atlas_small_fine_tune':'ATLAS Top tagging (4M) OmniLearn',
    'atlas_small':'ATLAS Top tagging (4M)',
    
    'true':'SR data',
    'gen':'Generated Jets',

    't_truth':'Top quarks',
    't_gen':'Generated Top quarks',
    'q_truth':'Quarks',
    'q_gen':'Generated quarks',
    'g_truth':'Gluons',
    'g_gen':'Generated gluons',
    'z_truth':'Z bosons',
    'z_gen':'Generated Z bosons',
    'w_truth':'W bosons',
    'w_gen':'Generated W bosons',

    'eic_truth':'EIC Particles',
    'eic_gen':'Generated EIC Particles',
        
    'jetclass':'JetClass',

    'herwig': 'Data (Herwig)',
    'pythia':'Pythia',
    'pythia_unfolded_baseline':'Pythia Unfolded',
    'pythia_unfolded_fine_tune':'Pythia Unfolded OmniLearn',
    'pythia_reweighted_baseline':'Pythia Reweighted',
    'pythia_reweighted_fine_tune':'Pythia Reweighted OmniLearn',
    }



def plot(jet1,jet2,flav1,flav2,nplots,title,plot_folder,is_big,names):
    var_names = ['Jet p$_{T}$ [GeV]', 'Jet $\eta$','Jet Mass [GeV]','Multiplicity']
    if nplots ==3:
        var_names = ['$\eta_{rel}$', '$\phi_{rel}$', 'log($1 - p_{Trel}$)']
    
    for ivar in range(nplots):        
        for i,unique in enumerate(np.unique(np.argmax(flav1,-1))):
            mask1 = np.argmax(flav1,-1)== unique
            mask2 = np.argmax(flav2,-1)== unique        
            
            feed_dict = {
                '{}_truth'.format(names[unique]):jet1[:,ivar][mask1],
                '{}_gen'.format(names[unique]):  jet2[:,ivar][mask2]
            }
            
            if i == 0:                            
                fig,gs,binning = HistRoutine(feed_dict,xlabel=var_names[ivar],
                                             plot_ratio=False,
                                             reference_name='{}_truth'.format(names[unique]),
                                             ylabel= 'Normalized entries')
            else:
                fig,gs,_ = HistRoutine(feed_dict,xlabel="{}".format(i),
                                       reference_name='{}_truth'.format(names[unique]),
                                       plot_ratio=False,
                                       fig=fig,gs=gs,binning=binning,
                                       ylabel= 'Normalized entries')
            ax0 = plt.subplot(gs[0])     
        fig.savefig('{}/Jetnet_{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')



def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    # import mplhep as hep
    # hep.set_style(hep.style.CMS)    
    # hep.style.use("CMS") 

def SetGrid(ratio=True,figsize=(9, 9),horizontal=False,npanels = 3):
    fig = plt.figure(figsize=figsize)
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    elif horizontal:
        gs = gridspec.GridSpec(1, npanels) 
        gs.update(wspace=0.0, hspace=0.025)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs

        
def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='gen',plot_ratio = False, plot_min=False):
    if plot_ratio:
        assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):
        ax0.plot(range(1,len(feed_dict[plot])+1),feed_dict[plot],
                 label=name_translate[plot],linewidth=2,linestyle=line_style[plot],color=colors[plot])
        if reference_name!=plot and plot_ratio:
            ratio = 100*np.divide(feed_dict[reference_name] -feed_dict[plot],feed_dict[reference_name])
            ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])
        if plot_min:
            min_val = np.min(feed_dict[plot])
            plt.axhline(y=min_val,linewidth=2,linestyle=line_style[plot],color=colors[plot])

    ax0.legend(loc='best',fontsize=18,ncol=1)            
    if plot_ratio:        
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
        plt.ylabel('Difference. (%)')
        plt.xlabel(xlabel)
        plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
        plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([-100,100])

    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0)    
        
    return fig,ax0


def FormatFig(xlabel,ylabel,ax0):
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

def HistRoutine(feed_dict,
                xlabel='',
                ylabel='Normalized entries',
                reference_name='data',
                logy=False,logx=False,
                binning=None,
                fig = None, gs = None,
                label_loc='best',
                plot_ratio=False,
                weights=None,
                uncertainty=None):
    if plot_ratio:
        assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    ref_plot = {'histtype':'stepfilled','alpha':0.2}
    other_plots = {'histtype':'step','linewidth':2}
    
    if fig is None:    
        fig,gs = SetGrid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])

    if plot_ratio:
        plt.tick_params(axis='x', labelbottom=False)
        ax1 = plt.subplot(gs[1],sharex=ax0)

        
    if binning is None:
        min_x = np.quantile(feed_dict[reference_name],0.001)
        max_x = np.quantile(feed_dict[reference_name],0.999)
        if min_x==max_x:
            max_x = np.quantile(feed_dict[reference_name],1.0)
        binning = np.linspace(min_x,max_x,50)

        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]

    
    if reference_name in feed_dict.keys():
        if weights is not None:
            reference_hist,_ = np.histogram(feed_dict[reference_name],weights=weights[reference_name],bins=binning,density=True)
        else:
            reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
            

    maxy = 0    
    for ip,plot in enumerate(feed_dict.keys()):
        plot_style = ref_plot if reference_name == plot else other_plots
        if weights is not None:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=name_translate[plot],
                              color=colors[plot],density=True,weights=weights[plot],**plot_style)
        else:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=name_translate[plot],
                              color=colors[plot],density=True,**plot_style)

        if np.max(dist) > maxy:
            maxy = np.max(dist)
            
        if plot_ratio:
            if reference_name!=plot:
                ratio = np.ma.divide(dist,reference_hist).filled(0)
                ax1.plot(xaxis,ratio,color=colors[plot],
                         marker='o',ms=10,
                         lw=0,markerfacecolor='none',markeredgewidth=3)
                if uncertainty is not None:
                    for ibin in range(len(binning)-1):
                        xup = binning[ibin+1]
                        xlow = binning[ibin]
                        ax1.fill_between(np.array([xlow,xup]),
                                         uncertainty[ibin],-uncertainty[ibin], alpha=0.3,color='k')    
    if logy:
        ax0.set_yscale('log')        
        ax0.set_ylim(1e-3,10*maxy)
        
    else:
        ax0.set_ylim(0,1.3*maxy)

    if logx:
        ax0.set_xscale('log')

    ax0.legend(loc=label_loc,fontsize=16,ncol=2)
    #ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 
        plt.ylabel('Ratio to Truth')
        plt.axhline(y=1.0, color='r', linestyle='-',linewidth=1)
        plt.ylim([0.85,1.15])
        plt.xlabel(xlabel)
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 

    return fig,gs,binning





def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(save_file,data):
    with open(save_file,'w') as f:
        json.dump(data, f)


