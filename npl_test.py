## NLP FUNCTION IN THEANO

import theano
from theano import tensor as T
from theano.tensor.signal import conv
from theano.tensor.nlinalg import MatrixInverse

Im_1 = T.tensor4('Im_1')
Im_2 = T.tensor4('Im_2')
F1_coeff = T.vector('F1_coeff')
DN_coeff = T.vector('DN_coeff')
sigmas = T.vector('sigmas')
DMOSi = T.matrix('DMOSi')
D0 = T.matrix('DMOSi')
D5 = T.matrix('DMOSi')
exp_f = T.scalar('exp_f')
exp_s = T.scalar('exp_s')
exp_g = T.scalar('exp_g')

### GAMMAS

Im_1b = abs(Im_1) ** (1 / exp_g)
Im_2b = abs(Im_2) ** (1 / exp_g)

##############################################################################
######  FUNCTION IMAGES 1
##############################################################################

F1 = T.zeros((9, 9))
F1 = T.set_subtensor(F1[0, 4], F1_coeff[0])
F1 = T.set_subtensor(F1[1, 4], F1_coeff[1])
F1 = T.set_subtensor(F1[2, 4], F1_coeff[2])
F1 = T.set_subtensor(F1[3, 4], F1_coeff[3])
F1 = T.set_subtensor(F1[4, 4], F1_coeff[4])
F1 = T.set_subtensor(F1[5, 4], F1_coeff[3])
F1 = T.set_subtensor(F1[6, 4], F1_coeff[2])
F1 = T.set_subtensor(F1[7, 4], F1_coeff[1])
F1 = T.set_subtensor(F1[8, 4], F1_coeff[0])

F1 = T.dot(F1, F1.T)
F1 = abs(F1)
sss = T.sum(F1, axis=[0, 1])
F1 = F1 / sss

DN_filts = T.zeros((2, 1, 9, 9))

D_aux = T.zeros((9, 9))
D_aux = T.set_subtensor(D_aux[2:7, 2:7], D0)
DN_filts = T.set_subtensor(DN_filts[0, 0, :, :], D_aux)

D_aux = T.zeros((9, 9))
D_aux = T.set_subtensor(D_aux[2:7, 2:7], D5)
DN_filts = T.set_subtensor(DN_filts[1, 0, :, :], D_aux)

# pyr = []
odd = [0, 0]
J = Im_1b

### pyrG = downsample_s(J,F1)

# Reflect Padding

output_coefficents = []
output_coefficents_Lap = []
output_coefficents_A2 = []

for N_b in range(0, 5):
    #    N_b = 0
    SJ0 = J.shape[0]
    SJ2 = J.shape[2]
    SJ3 = J.shape[3]

    J_aux = T.zeros((SJ0, 1, SJ2 + 8, SJ3 + 8))

    # size of F1 hardcoded to be 9x9
    J_aux = T.set_subtensor(J_aux[:, :, 4:4 + SJ2:1, 4:4 + SJ3:1], J[:, :, :, :])
    J_aux = T.set_subtensor(J_aux[:, :, 3, 4:4 + SJ3:1], J[:, :, 0, :])
    J_aux = T.set_subtensor(J_aux[:, :, 2, 4:4 + SJ3:1], J[:, :, 1, :])
    J_aux = T.set_subtensor(J_aux[:, :, 1, 4:4 + SJ3:1], J[:, :, 2, :])
    J_aux = T.set_subtensor(J_aux[:, :, 0, 4:4 + SJ3:1], J[:, :, 3, :])
    J_aux = T.set_subtensor(J_aux[:, :, -4, 4:4 + SJ3:1], J[:, :, -1, :])
    J_aux = T.set_subtensor(J_aux[:, :, -3, 4:4 + SJ3:1], J[:, :, -2, :])
    J_aux = T.set_subtensor(J_aux[:, :, -2, 4:4 + SJ3:1], J[:, :, -3, :])
    J_aux = T.set_subtensor(J_aux[:, :, -1, 4:4 + SJ3:1], J[:, :, -4, :])
    J_aux = T.set_subtensor(J_aux[:, :, :, 3], J_aux[:, :, :, 4])
    J_aux = T.set_subtensor(J_aux[:, :, :, 2], J_aux[:, :, :, 5])
    J_aux = T.set_subtensor(J_aux[:, :, :, 1], J_aux[:, :, :, 6])
    J_aux = T.set_subtensor(J_aux[:, :, :, 0], J_aux[:, :, :, 7])
    J_aux = T.set_subtensor(J_aux[:, :, :, -4], J_aux[:, :, :, -5])
    J_aux = T.set_subtensor(J_aux[:, :, :, -3], J_aux[:, :, :, -6])
    J_aux = T.set_subtensor(J_aux[:, :, :, -2], J_aux[:, :, :, -7])
    J_aux = T.set_subtensor(J_aux[:, :, :, -1], J_aux[:, :, :, -8])

    # convolution

    F1_aux = F1
    F1_aux = F1[None, None, :, :]
    R = T.nnet.conv.conv2d(J_aux, F1_aux, border_mode='full')  # ,filter_shape=(1,fso1+1,fso2+1)

    l1 = F1.shape[0] - 1  # // 2
    l2 = F1.shape[1] - 1  # // 2
    R_aux = R[:, :, l1:-l1, l2:-l2]

    # subsampling
    pyrG = R_aux[:, :, ::2, ::2]

    ### pyr = J - upsample_s(pyrG,odd,F1)

    SJ0 = pyrG.shape[0]
    SJ2 = pyrG.shape[2]
    SJ3 = pyrG.shape[3]

    pyrG_aux = T.zeros((SJ0, 1, SJ2 + 2, SJ3 + 2))

    pyrG_aux = T.set_subtensor(pyrG_aux[:, :, 1:1 + SJ2:1, 1:1 + SJ3:1], pyrG[:, :, :, :])
    pyrG_aux = T.set_subtensor(pyrG_aux[:, :, 0, 1:1 + SJ3:1], pyrG[:, :, 0, :])
    pyrG_aux = T.set_subtensor(pyrG_aux[:, :, -1, 1:1 + SJ3:1], pyrG[:, :, -1, :])
    pyrG_aux = T.set_subtensor(pyrG_aux[:, :, :, 0], pyrG_aux[:, :, :, 1])
    pyrG_aux = T.set_subtensor(pyrG_aux[:, :, :, -1], pyrG_aux[:, :, :, -2])

    [aa1, bb2, r, c] = T.shape(pyrG_aux)
    [aa, bb, J1, J2] = T.shape(J)
    odd[0] = 2 * r - J1
    odd[1] = 2 * c - J2

    r = 2 * r
    c = 2 * c
    R2 = T.zeros((aa1, bb2, r, c))
    R2 = T.set_subtensor(R2[:, :, ::2, ::2], 4 * pyrG_aux)
    R2 = T.nnet.conv.conv2d(R2, F1_aux, border_mode='full')  # R, filter, mode='constant', cval=(0.0))
    R2 = R2[:, :, 6:r + 6 - odd[0], 6:c + 6 - odd[1]]
    Lap_dom = J - R2  #

    ######

    DN_f_aux = DN_filts[0, :, :, :]
    DN_f_aux = DN_f_aux[None, :, :, :]
    A2 = T.nnet.conv.conv2d(abs(Lap_dom), DN_f_aux, border_mode='full')  # ,filter_shape=(1,fso1+1,fso2+1)

    nps1 = T.shape(Lap_dom)[2]
    nps2 = T.shape(Lap_dom)[3]
    fso1 = 4
    fso2 = 4
    A2 = A2[:, :, fso1:nps1 + fso1, fso2:nps2 + fso2]

    DN_dom = Lap_dom / (A2 + sigmas[0])

    output_coefficents_A2.append(A2)
    output_coefficents_Lap.append(Lap_dom)
    output_coefficents.append(DN_dom)

    J = pyrG

### LOW PASS

N_b = 5
Lap_dom = pyrG

DN_f_aux = DN_filts[1, :, :, :]
DN_f_aux = DN_f_aux[None, :, :, :]
A2 = T.nnet.conv.conv2d(abs(Lap_dom), DN_f_aux, border_mode='full')  # ,filter_shape=(1,fso1+1,fso2+1)

nps1 = T.shape(Lap_dom)[2]
nps2 = T.shape(Lap_dom)[3]
fso1 = 4
fso2 = 4
A2 = A2[:, :, fso1:nps1 + fso1, fso2:nps2 + fso2]

DN_dom = Lap_dom / (A2 + sigmas[1])

output_coefficents_A2.append(A2)
output_coefficents_Lap.append(Lap_dom)
output_coefficents.append(DN_dom)

#####

DN_dom_0 = output_coefficents[0]
DN_dom_1 = output_coefficents[1]
DN_dom_2 = output_coefficents[2]
DN_dom_3 = output_coefficents[3]
DN_dom_4 = output_coefficents[4]
DN_dom_5 = output_coefficents[5]

#####

Lap_dom_0 = output_coefficents_Lap[0]
Lap_dom_1 = output_coefficents_Lap[1]
Lap_dom_2 = output_coefficents_Lap[2]
Lap_dom_3 = output_coefficents_Lap[3]
Lap_dom_4 = output_coefficents_Lap[4]
Lap_dom_5 = output_coefficents_Lap[5]

#####

A2_0 = output_coefficents_A2[0]
A2_1 = output_coefficents_A2[1]
A2_2 = output_coefficents_A2[2]
A2_3 = output_coefficents_A2[3]
A2_4 = output_coefficents_A2[4]
A2_5 = output_coefficents_A2[5]

##############################################################################
######  FUNCTION IMAGES 2
##############################################################################

# pyr = []
odd = [0, 0]
J = Im_2b

### pyrG = downsample_s(J,F1)

# Reflect Padding

output_coefficents = []
output_coefficents_Lap = []
output_coefficents_A2 = []

for N_b in range(0, 5):
    #    N_b = 0
    SJ0 = J.shape[0]
    SJ2 = J.shape[2]
    SJ3 = J.shape[3]

    J_aux = T.zeros((SJ0, 1, SJ2 + 8, SJ3 + 8))

    # size of F1 hardcoded to be 9x9
    J_aux = T.set_subtensor(J_aux[:, :, 4:4 + SJ2:1, 4:4 + SJ3:1], J[:, :, :, :])
    J_aux = T.set_subtensor(J_aux[:, :, 3, 4:4 + SJ3:1], J[:, :, 0, :])
    J_aux = T.set_subtensor(J_aux[:, :, 2, 4:4 + SJ3:1], J[:, :, 1, :])
    J_aux = T.set_subtensor(J_aux[:, :, 1, 4:4 + SJ3:1], J[:, :, 2, :])
    J_aux = T.set_subtensor(J_aux[:, :, 0, 4:4 + SJ3:1], J[:, :, 3, :])
    J_aux = T.set_subtensor(J_aux[:, :, -4, 4:4 + SJ3:1], J[:, :, -1, :])
    J_aux = T.set_subtensor(J_aux[:, :, -3, 4:4 + SJ3:1], J[:, :, -2, :])
    J_aux = T.set_subtensor(J_aux[:, :, -2, 4:4 + SJ3:1], J[:, :, -3, :])
    J_aux = T.set_subtensor(J_aux[:, :, -1, 4:4 + SJ3:1], J[:, :, -4, :])
    J_aux = T.set_subtensor(J_aux[:, :, :, 3], J_aux[:, :, :, 4])
    J_aux = T.set_subtensor(J_aux[:, :, :, 2], J_aux[:, :, :, 5])
    J_aux = T.set_subtensor(J_aux[:, :, :, 1], J_aux[:, :, :, 6])
    J_aux = T.set_subtensor(J_aux[:, :, :, 0], J_aux[:, :, :, 7])
    J_aux = T.set_subtensor(J_aux[:, :, :, -4], J_aux[:, :, :, -5])
    J_aux = T.set_subtensor(J_aux[:, :, :, -3], J_aux[:, :, :, -6])
    J_aux = T.set_subtensor(J_aux[:, :, :, -2], J_aux[:, :, :, -7])
    J_aux = T.set_subtensor(J_aux[:, :, :, -1], J_aux[:, :, :, -8])

    # convolution

    F1_aux = F1
    F1_aux = F1[None, None, :, :]
    R = T.nnet.conv.conv2d(J_aux, F1_aux, border_mode='full')  # ,filter_shape=(1,fso1+1,fso2+1)

    l1 = F1.shape[0] - 1  # // 2
    l2 = F1.shape[1] - 1  # // 2
    R_aux = R[:, :, l1:-l1, l2:-l2]

    # subsampling
    pyrG = R_aux[:, :, ::2, ::2]

    ### pyr = J - upsample_s(pyrG,odd,F1)

    SJ0 = pyrG.shape[0]
    SJ2 = pyrG.shape[2]
    SJ3 = pyrG.shape[3]

    pyrG_aux = T.zeros((SJ0, 1, SJ2 + 2, SJ3 + 2))

    pyrG_aux = T.set_subtensor(pyrG_aux[:, :, 1:1 + SJ2:1, 1:1 + SJ3:1], pyrG[:, :, :, :])
    pyrG_aux = T.set_subtensor(pyrG_aux[:, :, 0, 1:1 + SJ3:1], pyrG[:, :, 0, :])
    pyrG_aux = T.set_subtensor(pyrG_aux[:, :, -1, 1:1 + SJ3:1], pyrG[:, :, -1, :])
    pyrG_aux = T.set_subtensor(pyrG_aux[:, :, :, 0], pyrG_aux[:, :, :, 1])
    pyrG_aux = T.set_subtensor(pyrG_aux[:, :, :, -1], pyrG_aux[:, :, :, -2])

    [aa1, bb2, r, c] = T.shape(pyrG_aux)
    [aa, bb, J1, J2] = T.shape(J)
    odd[0] = 2 * r - J1
    odd[1] = 2 * c - J2

    r = 2 * r
    c = 2 * c
    R2 = T.zeros((aa1, bb2, r, c))
    R2 = T.set_subtensor(R2[:, :, ::2, ::2], 4 * pyrG_aux)
    R2 = T.nnet.conv.conv2d(R2, F1_aux, border_mode='full')  # R, filter, mode='constant', cval=(0.0))
    R2 = R2[:, :, 6:r + 6 - odd[0], 6:c + 6 - odd[1]]
    Lap_dom = J - R2  #

    ######

    DN_f_aux = DN_filts[0, :, :, :]
    DN_f_aux = DN_f_aux[None, :, :, :]
    A2 = T.nnet.conv.conv2d(abs(Lap_dom), DN_f_aux, border_mode='full')  # ,filter_shape=(1,fso1+1,fso2+1)

    nps1 = T.shape(Lap_dom)[2]
    nps2 = T.shape(Lap_dom)[3]
    fso1 = 4
    fso2 = 4
    A2 = A2[:, :, fso1:nps1 + fso1, fso2:nps2 + fso2]

    DN_dom = Lap_dom / (A2 + sigmas[0])

    output_coefficents_A2.append(A2)
    output_coefficents_Lap.append(Lap_dom)
    output_coefficents.append(DN_dom)

    J = pyrG

### LOW PASS

N_b = 5
Lap_dom = pyrG

DN_f_aux = DN_filts[1, :, :, :]
DN_f_aux = DN_f_aux[None, :, :, :]
A2 = T.nnet.conv.conv2d(abs(Lap_dom), DN_f_aux, border_mode='full')  # ,filter_shape=(1,fso1+1,fso2+1)

nps1 = T.shape(Lap_dom)[2]
nps2 = T.shape(Lap_dom)[3]
fso1 = 4
fso2 = 4
A2 = A2[:, :, fso1:nps1 + fso1, fso2:nps2 + fso2]

DN_dom = Lap_dom / (A2 + sigmas[1])

output_coefficents_A2.append(A2)
output_coefficents_Lap.append(Lap_dom)
output_coefficents.append(DN_dom)

#####

DN_dom_0_2 = output_coefficents[0]
DN_dom_1_2 = output_coefficents[1]
DN_dom_2_2 = output_coefficents[2]
DN_dom_3_2 = output_coefficents[3]
DN_dom_4_2 = output_coefficents[4]
DN_dom_5_2 = output_coefficents[5]

#####

Lap_dom_0_2 = output_coefficents_Lap[0]
Lap_dom_1_2 = output_coefficents_Lap[1]
Lap_dom_2_2 = output_coefficents_Lap[2]
Lap_dom_3_2 = output_coefficents_Lap[3]
Lap_dom_4_2 = output_coefficents_Lap[4]
Lap_dom_5_2 = output_coefficents_Lap[5]

#####

A2_0_2 = output_coefficents_A2[0]
A2_1_2 = output_coefficents_A2[1]
A2_2_2 = output_coefficents_A2[2]
A2_3_2 = output_coefficents_A2[3]
A2_4_2 = output_coefficents_A2[4]
A2_5_2 = output_coefficents_A2[5]

#############################################


DD_0 = (T.mean(abs(DN_dom_0 - DN_dom_0_2) ** exp_s, axis=[2, 3]) ** (1 / exp_s))
DD_1 = (T.mean(abs(DN_dom_1 - DN_dom_1_2) ** exp_s, axis=[2, 3]) ** (1 / exp_s))
DD_2 = (T.mean(abs(DN_dom_2 - DN_dom_2_2) ** exp_s, axis=[2, 3]) ** (1 / exp_s))
DD_3 = (T.mean(abs(DN_dom_3 - DN_dom_3_2) ** exp_s, axis=[2, 3]) ** (1 / exp_s))
DD_4 = (T.mean(abs(DN_dom_4 - DN_dom_4_2) ** exp_s, axis=[2, 3]) ** (1 / exp_s))
DD_5 = (T.mean(abs(DN_dom_5 - DN_dom_5_2) ** exp_s, axis=[2, 3]) ** (1 / exp_s))

NLP_dist = ((DD_0 ** exp_f + DD_1 ** exp_f + DD_2 ** exp_f + DD_3 ** exp_f + DD_4 ** exp_f + DD_5 ** exp_f) / 6) ** (
            1 / exp_f)

#########################################


DMOSi_aux = DMOSi[:, 0]
NLP_disti_aux = NLP_dist[:, 0]

mm_D = T.mean(DMOSi_aux, axis=[0])
mm_N = T.mean(NLP_disti_aux, axis=[0])

DMOS_m = (DMOSi_aux - mm_D)
NLP_disti_m = (NLP_disti_aux - mm_N)

SS_DMOS = (T.mean(DMOS_m ** 2, axis=[0])) ** 0.5
SS_NLP_disti = (T.mean(NLP_disti_m ** 2, axis=[0])) ** 0.5

aux = (SS_DMOS * SS_NLP_disti);
rrr = T.mean(DMOS_m * NLP_disti_m, axis=[0]) / aux

## Derivatives

# P_Im_2 = theano.grad(rrr,Im_2)
# P_exp_f = theano.grad(rrr,exp_f)
# P_exp_s = theano.grad(rrr,exp_s)
# P_sigmas = theano.grad(rrr,sigmas)
# P_F1_coeff = theano.grad(rrr,F1_coeff)
# P_DN_coeff = theano.grad(rrr,DN_coeff)
# P_exp_g = theano.grad(rrr,exp_g)

# NLP_theano = theano.function(
#    [Im_1,Im_2,F1_coeff,D0,D5,sigmas,exp_s,exp_f,exp_g,DMOSi],
#    [rrr,P_Im_2,P_sigmas,P_F1_coeff,P_exp_g,P_exp_f,P_exp_s,P_DN_coeff,NLP_dist],
# )

## Without derivatives
NLP_theano = theano.function(
    [Im_1, Im_2, F1_coeff, D0, D5, sigmas, exp_s, exp_f, exp_g],
    [NLP_dist],
)
