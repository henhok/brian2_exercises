from adexfit_b2mofi import MarkramStepInjectionTraces
import efel
import numpy as np
import pandas as pd

current_steps = [-0.247559, 0.55425, 0.6004375, 0.646625]  # *nA
test_target = MarkramStepInjectionTraces('bbp_traces/L5_TTPC1_cADpyr232_4/hoc_recordings/',
                                         'soma_voltage_step', current_steps)

excluded_features = [ 'AP_phaseslope', 'AP_phaseslope_AIS',  'BAC_maximum_voltage', 'BAC_width', 'BPAPAmplitudeLoc1',
'BPAPAmplitudeLoc2', 'BPAPHeightLoc1', 'BPAPHeightLoc2', 'BPAPatt2', 'BPAPatt3', 'E10', 'E11', 'E12', 'E13',
'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E2', 'E20', 'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E3',
'E39', 'E39_cod', 'E4', 'E40', 'E5', 'E6', 'E7', 'E8', 'E9', 'check_AISInitiation',
'ohmic_input_resistance', 'ohmic_input_resistance_vb_ssse', 'sag_amplitude', 'sag_ratio1', 'sag_ratio2',
'time', 'voltage']

all_stim_features = np.setdiff1d(efel.getFeatureNames(), excluded_features)
a = test_target.getTargetValues(all_stim_features)
pd.DataFrame(a).to_json('allfeatures_test.json', orient='index')