# -*- coding: utf-8 -*-

from pyAbaqus import *
#########################################################
# Paramètres à changer                                  #
#########################################################

# Paramètres de la simulation
model_name = "Bicouche2D"
part1 = "AgilusLayer"
part2 = "VeroLayer"
initialTemp = 298.15 #Temperature initiale
FinalTemp = 353.15 #Temperature finale
JobName = "Bicouche"
step_name1 = "Heating"
step_name2 = "Cooling"

#les Dimensions:
h1 = 0.5E-3 #épaisseur du premiere couche en m
h2 = 0.5E-3 #épaisseur du deuxieme couche en m
L = 60E-3 #Longueur en m
startingPointX = 0 #Coordonnée en x axis du premier point
stratingPointY = 0 #Coordonnée en y axis du premier point

# Les proprietes des matériaux
ResAg30 = 2.439 #Contrainte résiduelle du premier matériau en %
ResVC = 0 #Contrainte résiduelle du deuxieme matériau en %
ExpansionAg30 = 2.3E-4
materials = {
                'Agilus30': {
                    'Density': 1140,#kg/m3
                    'Expansion': [(ExpansionAg30 + (ResAg30/100)/(FinalTemp-initialTemp),)], # unit 1/°C
                    'TempDepExpansion' : ON,
                    'Conductivity': [(0.0959,293.15),(0.2085,353.15)], # unit W/m.K
                    'TempDepConductivity' : ON,
                    'SpecificHeat': [(3373.68,293.15),(1570.18,353.15)], # unit J/Kg.K
                    'TempDepSpecificHeat' : ON, 
                },
                'VeroClear': {
                    'Density': 1190,#kg/m3
                    'Expansion': [(1.7E-4 + (ResVC/100)/(FinalTemp-initialTemp),)], # unit 1/°C
                    'TempDepExpansion' : OFF,
                    'Conductivity': [(0.1932,293.15),(0.1869,353.15)], # unit W/mm.K
                    'TempDepConductivity' : ON,
                    'SpecificHeat': [(1396,293.15),(1854,353.15)], # unit J/Kg.K
                    'TempDepSpecificHeat' : ON, 
                },
            }
Ag30Elastic = [(0.6e6, 0.45)] #Module de Young en MPa et coefficient de poisson

## Proprieté viscoelastique pour le VeroClear
VCElastic = [(2898750654.67,0.37)] # en Pascal
g = [
    0.0514, 0.0414, 0.0455, 0.0507, 0.0573, 0.0652, 0.0718, 0.0866, 
    0.0614, 0.0497, 0.0521, 0.0562, 0.0559, 0.0521, 0.0490, 0.0386, 
    0.0486, 0.0283, 0.0182, 0.0043, 0.0097, 0.0006, 0.0017, 0.0004, 
    0.0005, 6.30e-08, 0.0009
]
tau = [
    2.00E-08, 4.27E-07, 5.47E-06, 5.89E-05, 0.000547, 0.004524, 0.032439, 0.2, 
    1, 3.250259, 9.451896, 30.23741, 100, 315.2367, 927.9366, 8849.219, 2849.202, 
    25294.7, 7.29E+04, 653520.3, 2.13E+05, 5.37E+06, 2000000, 8.54E+07, 2.00E+07, 
    3.61E+08, 2.00E+09
]
VCprony = [(g_i, 0, tau_i) for g_i, tau_i in zip(g, tau)]
Tref = 295.15
C1 = 17.4
C2 = 66.35
AFcK = 24000.0
#Module de young en fonction de la temperature en Pascal de VeroClear
temperature = [25, 34.84471788322107, 39.9130471703688, 44.88199150521375, 49.950313210354125, 
               59.88820188004403, 69.82609054973392, 79.86335658971925, 90]
values = [1410.976769579425, 575.3672989698173, 313.7404490365931, 177.68751994883573, 98.12271762416792, 
          35.7118277534068, 16.111385439036535, 11.600357365671826, 8.785337205449062, 8.897035438964885]
values = [v*10**6 for v in values]
mu = 0.37
elasticVero = [(v*(10**6), mu, t) for v, t in zip(values, temperature)]


#########################################################
#                    Pre-Processing                     #
#########################################################

#Creation du modele
mdb.Model(name=model_name)

#Creation du premier Layer
ptLayer1 = [(startingPointX,stratingPointY),(L,h1)]
createRectangle2D(model_name, part1,ptLayer1[0],ptLayer1[1])

#Creation du deuxieme Layer
ptLayer2 = [(startingPointX,h1),(L,h1+h2)]
createRectangle2D(model_name, part2,ptLayer2[0],ptLayer2[1])

#Les proprietes generale tels que : la densité , la conductivité thermique, la capacité thermique et le coef d'expansion thermique
generalPropreties(model_name, materials)

# Modèle elastique pour l'Agilus30
elasticModel(model_name, "Agilus30", Ag30Elastic, OFF)

# SET UEXPAN for Agilus30
ExpansionUEXPAN(model_name,"Agilus30",1)

# Modèle viscoelastique pour le VeroClear + UTRS
viscoElastoModel(model_name, "VeroClear", VCElastic, VCprony)

# Modèle élastique pour le VeroClear
#elasticModel(model_name, "VeroClear", elasticVero, ON)

#Affectation des matériaux aux parts
Assign(model_name,part1,"Agilus30")
Assign(model_name,part2,"VeroClear")

#partition
partition2D(model_name, part1,YZPLANE,ptLayer1[1][0]/2)
partition2D(model_name, part2,YZPLANE,ptLayer2[1][0]/2)

#Assemblage
assemble_all_parts(model_name)

#Tie Constraint
tie_constraint_y_coord_by_part(model_name, ptLayer1[1][1], [part1,part2])

#Définition du type d'analyse
coupled_temp_disp_step(model_name, step_name1, 50, 1e-12, 0.1, 100000000, "Initial",0.01, 1,description='')
coupled_temp_disp_step(model_name, step_name2, 40, 1e-12, 0.1, 100000000, step_name1,0.01, 1,description='')
FieldOutput(model_name,step_name1,"Field1","History1")

#Temperature initial de la structure
predefined_temperature(model_name, "initialTemp",initialTemp)

#Condition aux limites
applyDispBC(model_name, part1,(ptLayer1[1][0]/2, 0, 0), [SET,SET,UNSET,UNSET,UNSET,SET],"bc1",entity_type='vertex')
edge = select_edge(model_name, part2, (ptLayer2[1][0]/2, ptLayer2[1][1], 0))
applyDispBC(model_name, part2,edge.pointOn[0], [SET,UNSET,UNSET,UNSET,UNSET,SET],"bc2",entity_type='edge')

#Temperature
edges = select_edges_with_fixed_y(model_name, part1, 0, tolerance=1e-6)
Amplitude = ((0.0, 0.844258814), (50.0, 1.0)) #Amplitude
tabularAmp(model_name,Amplitude,"Amp") #sert à creer une amplitude

apply_temperature_bc(model_name, part1,step_name1, FinalTemp,"Amp",edges[0],"a1")
apply_temperature_bc(model_name, part1,step_name1, FinalTemp,"Amp",edges[1],"b1")
modify_temp(model_name,"a1",step_name2,initialTemp,"")
modify_temp(model_name,"b1",step_name2,initialTemp,"")
convection(model_name, step_name2, initialTemp, 10, coefAmpl="", tempAmpl="")

#Mesh
mesh_func(model_name, part1, 1e-3, analysis_type='coupled_temp_disp', dimension='2D')
mesh_func(model_name, part2, 1e-3, analysis_type='coupled_temp_disp', dimension='2D')

#Job
jobSubmit(model_name, JobName, numCpus=1, numGPUs=0, memory=90, userSubroutine="utrs.for", precision=SINGLE)

#########################################################
#                    Post-Processing                    #
#########################################################
odb_path = os.path.join(os.getcwd(), JobName + ".odb")
instance_name = part1 + "_inst"
tolerance = 1e-5
pas = 1
FinalTemp = 80
initialTemp = 25
file_path = 'bending_angle_data.txt'
result1 = bending_angle_temp_range(odb_path, file_path,model_name,step_name1, instance_name, ptLayer1,FinalTemp,initialTemp,pas,tolerance)
a,b=bending_angle_for_frame(odb_path, step_name2, instance_name.upper(), 5, 93, -1, tolerance=0.01)
with open(file_path, 'a') as file:
    file.write(str(a) + ',' + str(b))
result2 = get_deformation_for_frame(odb_path, 'deformation_data1.txt', step_name1 , instance_name.upper(), ptLayer1[1][1], 353.15, tolerance=1e-5)
result3 = get_deformation_for_frame(odb_path, 'deformation_data2.txt', step_name2 , instance_name.upper(), ptLayer1[1][1], 298.15, tolerance=1e-5)
