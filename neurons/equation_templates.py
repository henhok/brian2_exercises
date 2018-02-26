# -*- coding: utf-8 -*-
__author__ = 'Andalibi, V., Hokkanen H., Vanni, S.'

'''
The preliminary version of this software has been developed at Aalto University 2012-2015, 
and the full version at the University of Helsinki 2013-2017. The software is distributed 
under the terms of the GNU General Public License. 
Copyright 2017 Vafa Andalibi, Henri Hokkanen and Simo Vanni.
'''

# This is version 2018-02-22 from CxSystem

from string import Template
import brian2 as br2

class EquationHelper(object):
    """
    Helper class for switching swiftly between neuron/synaptic current/synapse models in CxSystem.
    This is a preliminary version.

    Builds equation systems of the form:
    dvm/dt = ((I_NEURON_MODEL + I_SYNAPTIC + I_OTHER)/C) + I_VM_NOISE
    NEURON_MODEL_DYNAMICS
    SYNAPTIC_CURRENT_DYNAMICS
    """

    membrane_eq_template = '''
    dvm/dt = (($I_NEURON_MODEL + $I_SYNAPTIC_EXC + $I_SYNAPTIC_INH $I_TONIC)/C) $VM_NOISE : volt $BRIAN2_FLAGS
    $NEURON_MODEL_EQ
    $SYNAPTIC_EXC_EQ
    $SYNAPTIC_INH_EQ
    $TONIC_EQ
    '''

    # <editor-fold desc="NEURON MODEL COMPONENTS">
    # General strings will be added to all models
    default_soma_strings = {'I_TONIC': '+ tonic_current*(1-exp(-t/(50*msecond)))',
                            'TONIC_EQ': '',
                            'VM_NOISE': '+ noise_sigma*xi*taum_soma**-0.5',
                            'BRIAN2_FLAGS': '(unless refractory)'}
    default_dendrite_strings = {'I_TONIC': '',
                                'TONIC_EQ': '',
                                'VM_NOISE': '',
                                'BRIAN2_FLAGS': ''}   # Be careful! "Unless refractory" in dendrites will not cause an error, but behavior is WRONG


    NeuronModels = dict()
    NeuronModels['EIF'] = dict()
    NeuronModels['EIF'] = {'I_NEURON_MODEL': 'gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT)',
                           'NEURON_MODEL_EQ': ''}
    NeuronModels['EIF_PC'] = dict()
    NeuronModels['EIF_PC'] = {'soma': {'I_NEURON_MODEL': 'gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) + I_dendr',
                                       'NEURON_MODEL_EQ': ''},

                              'dendrite': {'I_NEURON_MODEL': 'gL*(EL-vm) + I_dendr',
                                           'NEURON_MODEL_EQ': ''}
                              }

    NeuronModels['ADEX'] = dict()
    NeuronModels['ADEX'] = {'I_NEURON_MODEL': 'gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) - w',
                            'NEURON_MODEL_EQ': 'dw/dt = (-a*(EL-vm)-w)/tau_w : amp'}
    NeuronModels['ADEX_PC'] = dict()
    NeuronModels['ADEX_PC'] = {'soma': {'I_NEURON_MODEL': 'gL*(EL-vm) + gL * DeltaT * exp((vm-VT) / DeltaT) + I_dendr',
                                        'NEURON_MODEL_EQ': ''},

                              'dendrite': {'I_NEURON_MODEL': 'gL*(EL-vm) + I_dendr',
                                           'NEURON_MODEL_EQ': ''}
                               }
    #</editor-fold>

    # <editor-fold desc="SYNAPTIC EXC/INH MODEL COMPONENTS">
    default_synaptic_excinh_strings = {}

    SynapticExcInhModels = dict()
    # Simple synaptic current models
    SynapticExcInhModels['SIMPLE_E'] = {'I_SYNAPTIC_EXC': 'ge * (Ee - vm)',
                                        'SYNAPTIC_EXC_EQ': 'dge/dt = -ge/tau_e : siemens'}
    SynapticExcInhModels['SIMPLE_I'] = {'I_SYNAPTIC_INH': 'gi * (Ei - vm)',
                                        'SYNAPTIC_INH_EQ': 'dgi/dt = -gi/tau_i : siemens'}

    # Simple alpha synapses
    SynapticExcInhModels['E_ALPHA'] = {'I_SYNAPTIC_EXC': 'gealpha * (Ee - vm)',
                                       'SYNAPTIC_EXC_EQ':
                                           '''dge/dt = -ge/tau_e_alpha : siemens
                                              dgealpha/dt = (ge-gealpha)/tau_e_alpha : siemens'''}

    SynapticExcInhModels['I_ALPHA'] = {'I_SYNAPTIC_INH': 'gialpha * (Ei - vm)',
                                       'SYNAPTIC_INH_EQ':
                                           '''dgi/dt = -gi/tau_i_alpha : siemens
                                              dgialpha/dt = (gi-gialpha)/tau_i_alpha : siemens'''}

    # Single-exponential AMPA+NMDA and GABA-A+GABA-B
    SynapticExcInhModels['AMPA_NMDA'] = {'I_SYNAPTIC_EXC': 'g_ampa*(E_ampa - vm) + g_nmda*B*(E_nmda - vm)',
                                         'SYNAPTIC_EXC_EQ':
                                             '''dg_ampa/dt = -g_ampa/tau_decay_ampa : siemens
                                                dg_nmda/dt = -g_nmda/tau_decay_nmda : siemens
                                                B = (1/(1+exp(-62*(vm/volt))*(1/3.57)))'''}

    SynapticExcInhModels['GABAA_GABAB'] = {'I_SYNAPTIC_INH': 'g_gabaa*(E_gabaa - vm) + g_gabab*(E_gabab - vm)',
                                           'SYNAPTIC_INH_EQ':
                                           '''dg_gabaa/dt = -g_gabaa/tau_decay_gabaa : siemens
                                              dg_gabab/dt = -g_gabab/tau_decay_gabab : siemens'''}

    # Dual exponential rise/decay for AMPA+NMDA and GABA-A+GABA-B
    SynapticExcInhModels['AMPA_NMDA_DUALEXP'] = {'I_SYNAPTIC_EXC': 'g_ampa_alpha*(E_ampa - vm) + g_nmda_alpha*B*(E_nmda - vm)',
                                                 'SYNAPTIC_EXC_EQ':
                                                     '''dg_ampa/dt = -g_ampa/tau_decay_ampa : siemens
                                                        dg_nmda/dt = -g_nmda/tau_decay_nmda : siemens
                                                        dg_ampa_alpha/dt = (g_ampa - g_ampa_alpha)/tau_rise_ampa : siemens
                                                        dg_nmda_alpha/dt = (g_nmda - g_nmda_alpha)/tau_rise_nmda : siemens
                                                        B = (1/(1+exp(-62*(vm/volt))*(1/3.57)))'''}

    SynapticExcInhModels['GABAA_GABAB_DUALEXP'] = {'I_SYNAPTIC_INH': 'g_gabaa*(E_gabaa - vm) + g_gabab*(E_gabab - vm)',
                                                   'SYNAPTIC_INH_EQ':
                                                       '''dg_gabaa/dt = -g_gabaa/tau_decay_gabaa : siemens
                                                          dg_gabab/dt = -g_gabab/tau_decay_gabab : siemens
                                                          dg_gabaa_alpha/dt = (g_gabaa - g_gabaa_alpha)/tau_rise_gabaa
                                                          dg_gabab_alpha/dt = (g_gabab - g_gabab_alpha)/tau_rise_gabab'''}



    # Variables that are compartment-specific; will have compartment name attached to them
    CompSpecificVariables = {'SIMPLE_E': ['ge'], 'SIMPLE_I': ['gi'],
                             'AMPA_NMDA': ['g_ampa', 'g_nmda'], 'GABAA_GABAB': ['g_gabaa', 'g_gabab'],
                             'E_ALPHA': ['ge', 'gealpha'], 'I_ALPHA': ['gi', 'gialpha']}

    # Receptors that need to be incremented in response to presynaptic spikes
    Receptors = {'SIMPLE_E': ['ge'], 'SIMPLE_I': ['gi'],
                             'AMPA_NMDA': ['g_ampa', 'g_nmda'], 'GABAA_GABAB': ['g_gabaa', 'g_gabab'],
                             'E_ALPHA': ['ge'], 'I_ALPHA': ['gi']}

    # </editor-fold>

    def __init__(self, neuron_model='EIF', is_pyramidal=False, compartment='soma', exc_model='SIMPLE_E', inh_model='SIMPLE_I', custom_strings=None):
        self.neuron_model = neuron_model
        self.is_pyramidal = is_pyramidal
        self.compartment = compartment
        self.exc_model = exc_model
        self.inh_model = inh_model
        self.custom_strings = custom_strings

        if compartment == 'soma':
            self.compartment_type = 'soma'
        else:
            self.compartment_type = 'dendrite'

        assert neuron_model in EquationHelper.NeuronModels.keys(), \
            "Neuron model must be defined in NeuronModels"
        assert exc_model in EquationHelper.SynapticExcInhModels.keys(), \
            "Excitation model must be defined in SynapticExcInhModels"
        assert inh_model in EquationHelper.SynapticExcInhModels.keys(), \
            "Inhibition model must be defined in SynapticExcInhModels"

        # Add neuron model specific keys&strings to default strings; overlapping definitions will be taken
        # from model specific dict
        if is_pyramidal is True:
            assert self.compartment_type in ['soma', 'dendrite'], "Compartment must be defined (soma or dendrite)"
            if self.compartment_type == 'soma':
                self.neuron_model_strings = dict(EquationHelper.default_soma_strings)
            else:
                self.neuron_model_strings = dict(EquationHelper.default_dendrite_strings)

            self.neuron_model_strings.update(EquationHelper.NeuronModels[neuron_model+'_PC'][self.compartment_type])
        else:
            self.neuron_model_strings = dict(EquationHelper.default_soma_strings)
            self.neuron_model_strings.update(EquationHelper.NeuronModels[neuron_model])

        # Add synaptic E/I model specific keys&strings to default strings
        # Pyramidal cells are assumed to have alpha synapses (non-zero rise time)
        self.synaptic_excinh_model_strings = dict(EquationHelper.default_synaptic_excinh_strings)
        self.synaptic_excinh_model_strings.update(EquationHelper.SynapticExcInhModels[exc_model])
        self.synaptic_excinh_model_strings.update(EquationHelper.SynapticExcInhModels[inh_model])

        # Aggregate all compartment-specific variables to a common list
        self.comp_specific_vars = EquationHelper.CompSpecificVariables[exc_model] + \
                                  EquationHelper.CompSpecificVariables[inh_model]

    def getMembraneEquation(self, return_string=False):

        membrane_equation = Template(EquationHelper.membrane_eq_template)
        all_membrane_model_strings = dict(self.neuron_model_strings)
        all_membrane_model_strings.update(self.synaptic_excinh_model_strings)
        if self.custom_strings is not None:
            all_membrane_model_strings.update(self.custom_strings)

        try:
            membrane_equation = membrane_equation.substitute(all_membrane_model_strings)
        except KeyError:
            print 'Undefined key in membrane equation'

        eq_lines = membrane_equation.splitlines()
        eq_lines = [line.strip()+'\n' for line in eq_lines if len(line.strip()) > 0]

        final_membrane_equation = ''.join(eq_lines)

        if return_string is True:
            return final_membrane_equation
        else:
            substitutables = {k: k+'_'+self.compartment for k in self.comp_specific_vars}
            compartment_eq = br2.Equations(final_membrane_equation, **substitutables)

            return compartment_eq


if __name__ == '__main__':
    x = EquationHelper(neuron_model='EIF', is_pyramidal=True, compartment='soma', exc_model='E_ALPHA')
    print x.getMembraneEquation(return_string=False)