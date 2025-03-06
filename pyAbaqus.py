# -*- coding: utf-8 -*-

from abaqus import *
from abaqusConstants import *
import __main__
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
from odbAccess import openOdb
import os
import itertools
import math



def createRectangle2D(model_name, part_name,pt1,pt2):
    '''
        This function creates a 2D rectangular part in the specified Abaqus model.
        Input:
                - model_name: The name of the model in which to create the part.
                - part_name: The name you want to assign to the new part.
                - pt1: Tuple representing the (X, Y) coordinates of the first corner of the rectangle.
                - pt2: Tuple representing the (X, Y) coordinates of the opposite corner of the rectangle.
        Output:
                - A 2D planar part is created in the specified model.
        Usage:
                createRectangle2D("Model-1", "Layer1", (0, 0), (60, 1))
    '''
    model = mdb.models[model_name]
    s = model.ConstrainedSketch(name='__profile__', sheetSize=10.0)
    s.rectangle(point1=pt1, point2=pt2)
    p = model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    p.BaseShell(sketch=s)



def partition2D(model_name, part_name,plane,offset):
    '''
        This function creates a 2D partition for a 2D part in the specified Abaqus model.
        Input:
                - model_name: The name of the model containing the part.
                - part_name: The name of the part you want to partition.
                - plane: The plane of the partition (e.g., XZPLANE, XYPLANE).
                - offset: The offset along the other axis where the partitioning plane will be created.
        Output:
                - The specified part is partitioned into two regions.
                
        Usage:
                partition2D("Model-1", "Layer1", XZPLANE, 1)
    '''
    p = mdb.models[model_name].parts[part_name]
    datum = p.DatumPlaneByPrincipalPlane(principalPlane=plane, offset=offset)
    p.PartitionFaceByDatumPlane(datumPlane=p.datums[datum.id], faces=p.faces)



def generalPropreties(model_name, materials):
    '''
        This function defines general material properties in the specified Abaqus model.
        Input:
                - model_name: The name of the model in which to define the materials.
                - materials: A dictionary where each key is a material name, and the value is another dictionary 
                  containing the material properties, including:
                    * Density
                    * Thermal Expansion with temperature dependency
                    * Conductivity with temperature dependency
                    * Specific Heat with temperature dependency
        Output:
                - Materials are defined in the model with the specified properties.
                
        Usage: 
                materials = {
                    'Material1': {
                        'Density': Value,
                        'Expansion': [(value, temp)] if 'TempDepExpansion' == ON else [(value,)],  # unit: 1/°C
                        'TempDepExpansion': ON/OFF, 
                        'Conductivity': [(value, temp)], 
                        'TempDepConductivity': ON/OFF,
                        'SpecificHeat': [(value, temp)], 
                        'TempDepSpecificHeat': ON/OFF, 
                    },
                    'Material2': {
                        # Define properties for Material2
                    },
                }
                generalPropreties("Model-1", materials)
    '''
    model = mdb.models[model_name]
    for material_name, properties in materials.items():
        # Check if the material is already created
        if material_name in model.materials.keys():
            mat = model.materials[material_name]
        # Create the material
        else:
            mat = model.Material(name=material_name)
        # Density
        mat.Density(table=((properties['Density'], ), ))
        # Thermal Expansion with temperature dependency
        mat.Expansion(
            table=properties['Expansion'], 
            temperatureDependency=properties['TempDepExpansion']
        )
        # Conductivity with temperature dependency
        mat.Conductivity(
                    temperatureDependency=properties['TempDepConductivity'],
                    #table=tuple(c for c in properties['Conductivity'])
                    table=properties['Conductivity']
                    )
        # Specific Heat with temperature dependency
        mat.SpecificHeat(
            temperatureDependency=properties['TempDepSpecificHeat'],
            table=properties['SpecificHeat']
        )
        # Create a homogeneous solid section for the material
        section_name = material_name
        model.HomogeneousSolidSection(name=section_name, material=material_name, thickness=None)



def ExpansionUEXPAN(model_name,material_name,num_depvar):
    del mdb.models[model_name].materials[material_name].expansion
    mdb.models[model_name].materials[material_name].Expansion(userSubroutine=ON)
    mdb.models[model_name].materials[material_name].Depvar(n=num_depvar)



def elasticModel(model_name, material_name, values, tempdep):
    '''
        This function defines the elastic properties for a material in the specified Abaqus model.
        Input:
                - model_name: The name of the model in which to define the material.
                - material_name: The name of the material to which the elastic properties will be assigned.
                - values: A list of tuples representing the elastic properties. 
                          Each tuple contains the Young's Modulus and Poisson's ratio, and optionally, temperature.
                - tempdep: A flag indicating whether the properties are temperature dependent (ON or OFF).
        Output:
                - Elastic material properties are assigned to the specified material in the model.
                
        Usage:
                In the case of temperature-dependent elastic modulus:
                    temperature = [temperature_1, temperature_2, ...]
                    values = [Young's Modulus_1, Young's Modulus_2, ...]
                    poisson_ratio = value
                    elastic_values = [(v, poisson_ratio, t) for v, t in zip(values, temperature)]
                    elasticModel("Model-1", material_name, elastic_values, ON)
                    
                In the case of non-temperature-dependent properties:
                    elasticModel("Model-1", material_name, [(Young's Modulus, Poisson's ratio)], OFF)
    '''
    model = mdb.models[model_name] 
    if material_name in model.materials.keys():
        mat = model.materials[material_name]
    else:
        mat = model.Material(name=material_name)
    mat.Elastic(temperatureDependency=tempdep, table=values)


def ogdenHyper(model_name,material_name,order,values):
    '''
        This function defines Ogden hyperelastic material properties in the specified Abaqus model.
        Input:
                - model_name: The name of the model in which to define the material.
                - material_name: The name of the material to which the Ogden hyperelastic properties will be assigned.
                - order: The order of the Ogden model (e.g., 2 for a second-order model).
                - values: A list of tuples representing the hyperelastic properties. 
                          Each tuple contains the parameters (mu1, alpha1,mu2, alpha2,..., D1, D2,...) for the Ogden model.
        Output:
                - Ogden hyperelastic material properties are assigned to the specified material in the model.
                
        Usage:
                OgdenOrder2Values = [(mu1, alpha1, mu2, alpha2, D1, D2)]
                order = 2
                ogdenHyper("Model-1", material_name, order, OgdenOrder2Values)
    '''
    model = mdb.models[model_name] 
    if material_name in model.materials.keys():
        mat = model.materials[material_name]
    else:
        mat = model.Material(name=material_name)
    mat.Hyperelastic(
        materialType=ISOTROPIC, testData=OFF, type=OGDEN, n=order, 
        volumetricResponse=VOLUMETRIC_DATA, table=values)


def viscoElastoModel(model_name, material_name, elastic_table, prony_table, 
               elastic_moduli_type=INSTANTANEOUS, 
               viscoelastic_domain=TIME, 
               viscoelastic_type=PRONY, 
               trs_definition=USER, 
               num_depvar=1):
    '''
    This function defines the viscoelastic material properties in the specified Abaqus model.

    Input:
        - model_name (str): The name of the Abaqus model in which the material properties will be defined.        
        - material_name (str): The name of the material to which the viscoelastic and elastic properties will be assigned.        
        - elastic_table (list of tuples): [(E1, nu1)]
        - prony_table (list of tuples): A list of tuples representing the Prony series parameters for viscoelastic modeling.
                                        Each tuple contains the values for (g, k, tau), where:
                                          - g: Shear modulus ratio.
                                          - k: Bulk modulus ratio.
                                          - tau: Relaxation time.
                                        Example: [(g1, k1, tau1), (g2, k2, tau2), ...]        
        - elastic_moduli_type (optional, default=INSTANTANEOUS): Specifies whether the elastic moduli are INSTANTANEOUS or LONG_TERM.
        - viscoelastic_domain (optional, default=TIME): The domain for viscoelastic behavior, typically TIME or FREQUENCY.
        - viscoelastic_type (optional, default=PRONY): Specifies the type of viscoelastic behavior, usually PRONY for Prony series.
        - trs_definition (optional, default=USER): Specifies the definition for time-temperature shift behavior, often USER or WLF (Williams-Landel-Ferry).
        - num_depvar (optional, default=1): The number of state-dependent variables (DEPVAR) needed for the material behavior in Abaqus.
    
    Output:
        - The specified viscoelastic and elastic material properties are assigned to the material in the model.
    '''
    
    model = mdb.models[model_name] 
    if material_name in model.materials.keys():
        mat = model.materials[material_name]
    else:
        mat = model.Material(name=material_name)
    mat.Elastic(
        moduli=elastic_moduli_type, 
        table=elastic_table
    )
    mat.Viscoelastic(
        domain=viscoelastic_domain, 
        time=viscoelastic_type, 
        table=prony_table
    )
    mat.viscoelastic.Trs(
        definition=trs_definition
    )
    mat.Depvar(n=num_depvar)



def assemble_all_parts(model_name):
    '''
        This function assembles all parts in the specified Abaqus model.
        Input:
                - model_name: The name of the model containing the parts to be assembled.
        Output:
                - All parts in the model are instantiated and assembled in the root assembly.
                - Name of the instance : part_name+_inst
        Usage:
                assemble_all_parts("Model-1")
    '''
    model = mdb.models[model_name]
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    for part_name, part in model.parts.items():
        instance_name = part_name + '_inst'
        assembly.Instance(name=instance_name, part=part, dependent=ON)





def tie_constraint(model_name):
    '''
        This function creates tie constraints for all pairs of part instances in the specified Abaqus model.
        It is used for 3D modeling to ensure that the surfaces of different parts remain connected during analysis.
        Input:
                - model_name: The name of the model containing the part instances to be tied.
        Output:
                - Tie constraints are created for all pairs of part instances in the model.
        Usage:
                tie_constraint("Model-1")
    '''
    model = mdb.models[model_name]
    assembly = model.rootAssembly
    regions = []
    for i,instanceName in enumerate(assembly.instances.keys()):
        instance = assembly.instances[instanceName]
        faces = instance.faces
        regions.append(assembly.Surface(side1Faces=faces, name='surf'+str(i)))
    combinations = itertools.combinations(regions, 2)
    combinations_list = list(combinations)
    i = 0
    for c in combinations_list:
        mdb.models[model_name].Tie(name='Constraint'+str(i), master=c[0], slave=c[1], 
                positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, 
                thickness=ON)
        i = i+1




def tie_constraint2D(model_name):
    '''
        This function creates tie constraints for all edges of part instances in a 2D model.
        It ensures that the edges of different parts are connected during the analysis.
        Input:
                - model_name: The name of the model containing the 2D part instances to be tied.
        Output:
                - Tie constraints are created for all edge pairs in the 2D part instances within the model.
        Usage:
                tie_constraint2D("Model-1")
    '''
    model = mdb.models[model_name]
    assembly = model.rootAssembly
    regions = []
    for i,instanceName in enumerate(assembly.instances.keys()):
        instance = assembly.instances[instanceName]
        edges  = instance.edges
        regions.append(assembly.Surface(side1Edges=edges, name='surf'+str(i)))
    combinations = itertools.combinations(regions, 2)
    combinations_list = list(combinations)
    i = 0
    for c in combinations_list:
        mdb.models[model_name].Tie(name='Constraint'+str(i), master=c[0], slave=c[1], 
        positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, 
        thickness=ON)
        i = i+1


def tie_constraint_y_coord_by_part(model_name, y_coord, part_names, tolerance=1e-6):
    '''
    This function creates tie constraints for edges in 2D part instances within the same y-coordinate range,
    for instances derived from the specified part names.
    
    Input:
        - model_name: The name of the model containing the 2D part instances.
        - y_coord: The fixed y-coordinate to match for creating the tie constraints.
        - part_names: A list of part names from which instances are derived for applying tie constraints.
        - tolerance: A small value to account for floating-point precision (default: 1e-6).
    
    Output:
        - Tie constraints are created for edge pairs in instances derived from the specified part names with matching y-coordinates.
    '''
    model = mdb.models[model_name]
    assembly = model.rootAssembly
    regions = []
    for instanceName, instance in assembly.instances.items():
        if instance.partName in part_names:  
            matching_edges = []
            for edge in instance.edges:
                edge_coords = edge.pointOn 
                for edge_coord in edge_coords:
                    if abs(edge_coord[1] - y_coord) < tolerance:
                        matching_edges.append(instance.edges.findAt((edge_coord,)))
                        break 
            if matching_edges:
                regions.append(matching_edges)
    if len(regions) < 2:
        print("Not enough matching edges found in the assembly to create tie constraints.")
        return
    for i, (edges1, edges2) in enumerate(itertools.combinations(regions, 2)):
        try:
            surface1 = assembly.Surface(side1Edges=edges1, name='master_surf_' + str(i))
            surface2 = assembly.Surface(side1Edges=edges2, name='slave_surf_' + str(i))
            
            model.Tie(name='Constraint_' + str(i), master=surface1, slave=surface2,
                      positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON,
                      thickness=ON)
        except Exception as e:
            print("Error creating tie constraint between surfaces: " + str(e))
    print(str(i+1) + " tie constraints created between part instances derived from the specified parts at y=" + str(y_coord) + ".")





def Assign(model_name, name, material):
    '''
    This function assigns a material to a part in the specified Abaqus model.
    It handles both 2D and 3D parts by checking if the part has cells (3D) or faces (2D).
    
    Input:
            - model_name: The name of the model containing the part.
            - part_name: The name of the part to which the material will be assigned.
            - material_name: The name of the material to assign to the part.
    
    Output:
            - The specified part is assigned the specified material.
    
    Usage:
            Assign("Model-1", "Part_Name", "Material_Name")
    '''
    model = mdb.models[model_name]
    part_name = name
    part = model.parts[part_name]
    name = part_name
    if part.cells:
        region = part.Set(cells=part.cells, name='Set_' + name)
    elif part.faces:
        region = part.Set(faces=part.faces, name='Set_' + name)
    else:
        raise ValueError("Part " + part_name + " is neither 2D nor 3D.")
    part.SectionAssignment(
    region=region,
    sectionName=material,
    offset=0.0,
    offsetType=MIDDLE_SURFACE,
    offsetField='',
    thicknessAssignment=FROM_SECTION
    )






def import_Catia(model_name,file_path):
    '''
        This function imports a CATIA file as a part into the specified Abaqus model.
        Input:
                - model_name: The name of the model into which the CATIA part will be imported.
                - file_path: The file path to the CATIA file (.CATPart) to be imported.
        Output:
                - The CATIA part is imported into the model as a new part.
        
        Usage:
                import_Catia("Model-1", "C:/path/to/your/file.CATPart")
    '''
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    catia = mdb.openCatia(
        fileName=file_path, 
        topology=SOLID, useServer=True)
    mdb.models[model_name].PartFromGeometryFile(name=base_name, geometryFile=catia, 
        combine=False, dimensionality=THREE_D, type=DEFORMABLE_BODY, scale=1.0)
    p = mdb.models[model_name].parts[base_name]

       
def importAndAssign(model_name,file_path,material):
    '''
        This function imports a CATIA part into the specified Abaqus model and assigns a material to it.
        Input:
                - model_name: The name of the model into which the CATIA part will be imported.
                - file_path: The file path to the CATIA file (.CATPart) to be imported.
                - material_name: The name of the material to assign to the imported part.
        Output:
                - The CATIA part is imported into the model and assigned the specified material.
        
        Usage:
                importAndAssign("Model-1", "C:/path/to/your/file.CATPart", "Material_Name")
    '''
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    catia = mdb.openCatia(
        fileName=file_path, 
        topology=SOLID, useServer=True)
    mdb.models[model_name].PartFromGeometryFile(name=base_name, geometryFile=catia, 
        combine=False, dimensionality=THREE_D, type=DEFORMABLE_BODY, scale=1.0)
    p = mdb.models[model_name].parts[base_name]
    Assign(model_name,base_name,material)

# Coupled Temp disp 
def coupled_temp_disp_step(model_name, step_name, time_period, min_inc, max_inc, max_num_inc, prev,initial_inc, deltmx,description=''):
    model = mdb.models[model_name]
    model.CoupledTempDisplacementStep(name=step_name, previous=prev, timePeriod=time_period,
                                      maxNumInc=max_num_inc, initialInc=initial_inc, minInc=min_inc,
                                      maxInc=max_inc, deltmx=int(deltmx),cetol=None, creepIntegration=IMPLICIT_EXPLICIT, nlgeom=ON,description=description)

# Statitc
def general_static(model_name,step_name,previous,timePeriod,initialInc,minInc,maxInc,nlgeom=ON):
    mdb.models[model_name].StaticStep(name=step_name, previous=previous, 
        timePeriod=timePeriod, initialInc=initialInc, minInc=minInc, maxInc=maxInc, 
        nlgeom=nlgeom)

def FieldOutput(model_name,step_name,FieldName,HistoryName):
    mdb.models[model_name].FieldOutputRequest(name='F-Output-2', createStepName=step_name, variables=('MISES', 'U', 'UT', 'UR', 'NT'))
    mdb.models[model_name].HistoryOutputRequest(name='H-Output-2', createStepName=step_name, variables=('FTEMP', 
                                                                                                        'HFLA', 'HTL', 'HTLA', 'RADFL', 
                                                                                                        'RADFLA', 'RADTL', 'RADTLA', 'VFTOT', 
                                                                                                        'SJD', 'SJDA', 'SJDT', 'SJDTA', 'WEIGHT'))




def predefined_temperature(model_name, field_name,initialTemp):
    '''
    This function applies a predefined temperature to all faces (for 2D parts) or cells (for 3D parts)
    in all instances of the assembly.
    
    Input:
        - model_name: The name of the model containing the assembly.
        - initialTemp: The temperature to be applied to all faces (2D) or cells (3D) of each instance.
    
    Output:
        - A predefined temperature field is applied to all faces (for 2D) or cells (for 3D) of all part instances.
    '''
    a = mdb.models[model_name].rootAssembly
    for instance_name in a.instances.keys():
        instance = a.instances[instance_name]
        if len(instance.faces) > 0:
            instance_faces = instance.faces[:]
            region = a.Set(faces=instance_faces, name='Faces_Set_' + instance_name)
            mdb.models[model_name].Temperature(
                name=field_name+'_'+instance_name, 
                createStepName='Initial', 
                region=region, 
                distributionType=UNIFORM, 
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, 
                magnitudes=(initialTemp,)
            )
        elif len(instance.cells) > 0:
            instance_cells = instance.cells[:]  # Collect all cells
            region = a.Set(cells=instance_cells, name='Cells_Set_' + instance_name)
            mdb.models[model_name].Temperature(
                name=field_name+'_'+instance_name, 
                createStepName='Initial', 
                region=region, 
                distributionType=UNIFORM, 
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, 
                magnitudes=(initialTemp,)
            )

def convection(model_name, step_name, temp, coef, coefAmpl="", tempAmpl=""):
    a = mdb.models[model_name].rootAssembly
    w = 1
    for instance_name in a.instances.keys():
        instance = a.instances[instance_name]
        instance_edges = instance.edges
        for i in instance_edges:
            region = a.Surface(side1Edges=instance_edges.findAt((i.pointOn[0],)), name='Convection-edge' + str(w))
            mdb.models[model_name].FilmCondition(
                name="film" + str(w), 
                createStepName=step_name, 
                surface=region, 
                definition=EMBEDDED_COEFF, 
                filmCoeff=coef, 
                filmCoeffAmplitude=coefAmpl, 
                sinkTemperature=temp, 
                sinkAmplitude=tempAmpl, 
                sinkDistributionType=UNIFORM, 
                sinkFieldName=''
            )
            w += 1




def apply_temperature_bc(model_name, part_name,step_name, appliedTemp,ampl,edge,name):
    model = mdb.models[model_name]
    assembly = mdb.models[model_name].rootAssembly
    inst = part_name+'_inst'
    e1 = assembly.instances[inst].edges
    edge1 = e1.findAt(edge.pointOn)
    assembly.Set(edges=edge1, name=name+'___')
    region = assembly.sets[name+'___']
    mdb.models[model_name].TemperatureBC(name=name, 
        createStepName=step_name, region=region, fixed=OFF, 
        distributionType=UNIFORM, fieldName='', magnitude=appliedTemp, 
        amplitude=ampl)

def tabularAmp(model_name,Amplitude,name):
    model = mdb.models[model_name]
    model.TabularAmplitude(name=name, timeSpan=STEP, data=Amplitude)

def select_bottom_face(model_name, part_name):
    '''
        This Function selects the face with the lowest z-coordinate (bottom face) of a part.
        Input:
                - The Model name
                - The name of the part
        Output:
                - The face with the lowest z-coordinate is returned
    '''
    part = mdb.models[model_name].parts[part_name]
    lowest_z = float('inf')
    bottom_face = None
    for face in part.faces:
        coord = face.pointOn
        z_coord = coord[0][2]
        if z_coord < lowest_z:
            lowest_z = z_coord
            bottom_face = face
    return bottom_face



import math

def select_edge(model_name, part_name, coord):
    '''
    This function returns the edge that contains or is closest to the given coordinates.
    
    Input:
        - model_name: The name of the model containing the part.
        - part_name: The name of the part containing the edges.
        - coord: A tuple of coordinates (x, y, z).
    
    Output:
        - Returns the edge closest to the input coordinates.
    '''
    part = mdb.models[model_name].parts[part_name]
    min_distance = float('inf')
    closest_edge = None
    for edge in part.edges:
        edge_coords = edge.pointOn
        for edge_coord in edge_coords:
            distance = math.sqrt((edge_coord[0] - coord[0]) ** 2 + 
                                 (edge_coord[1] - coord[1]) ** 2 + 
                                 (edge_coord[2] - coord[2]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_edge = edge
    return closest_edge


def select_edges_with_fixed_y(model_name, part_name, y_coord, tolerance=1e-6):
    '''
    This function returns all edges that have at least one point whose y-coordinate matches
    the given y-coordinate (within a specified tolerance).
    
    Input:
        - model_name: The name of the model containing the part.
        - part_name: The name of the part containing the edges.
        - y_coord: The fixed y-coordinate.
        - tolerance: A small value to account for floating-point precision (default: 1e-6).
    
    Output:
        - Returns a list of tuples, where each tuple contains:
            1. The edge.
            2. A list of coordinates for points on the edge.
    '''
    part = mdb.models[model_name].parts[part_name]
    matching_edges = []
    for edge in part.edges:
        edge_coords = edge.pointOn  # Returns the coordinates of points on the edge
        for edge_coord in edge_coords:
            # Check if the y-coordinate is within the specified tolerance
            if abs(edge_coord[1] - y_coord) < tolerance:
                # If it matches, store the edge and its coordinates
                matching_edges.append(edge)
                break  # Move to the next edge once one point on this edge matches
    return matching_edges









def applyDispBC(model_name, part_name, entity, bc,name, entity_type='face'):
    '''
        This function applies a displacement boundary condition to a specified face in the Abaqus model.
        Input:
                - model_name: The name of the model containing the part.
                - part_name: The name of the part containing the face.
                - face: The coordinates of the face where the boundary condition will be applied.
                - BC = [u1,u2,u3,ur1,ur2,ur3]
        Output:
                - A displacement boundary condition is applied to the specified face.
        
        Usage:
                applyDispBC("Model-1", "Part-1",(x, y, z), [SET,UNSET,UNSET,UNSET,UNSET,UNSET],entity_type='face')
    '''
    a = mdb.models[model_name].rootAssembly
    instance = a.instances[part_name + '_inst']
    if entity_type == 'face':
        if isinstance(entity, tuple):
            region = a.Set(faces=instance.faces.findAt((entity,)), name='Set-BC-' + part_name)
        else:
            region = a.Set(faces=instance.faces.findAt(entity.pointOn), name='Set-BC-' + part_name)
    elif entity_type == 'vertex':
        if isinstance(entity, tuple):
            region = a.Set(vertices=instance.vertices.findAt((entity,)), name='Set-BC-' + part_name)
        else:
            region = a.Set(vertices=instance.vertices.findAt(entity.pointOn), name='Set-BC-' + part_name)
    elif entity_type == 'edge':
        if isinstance(entity, tuple):
            region = a.Set(edges=instance.edges.findAt((entity,)), name='Set-BC-' + part_name)
        else:
            region = a.Set(edges=instance.edges.findAt(entity.pointOn), name='Set-BC-' + part_name)
    else:
        raise ValueError("Invalid entity_type provided. Must be 'face', 'vertex', or 'edge'.")
    mdb.models[model_name].DisplacementBC(name=name, createStepName='Initial', 
        region=region, u1=bc[0], u2=bc[1], u3=bc[2], ur1=bc[3], ur2=bc[4], ur3=bc[5], 
        amplitude=UNSET, distributionType=UNIFORM, fieldName='', 
        localCsys=None)


def modify_temp(model_name,name,step_name,temp,amplitude):
    mdb.models[model_name].boundaryConditions[name].setValuesInStep(
        stepName=step_name, magnitude=temp, amplitude=amplitude)


def mesh_func(model_name, part_name, size, analysis_type='static', dimension='3D'):
    '''
    This function generates a mesh for the specified part in the model.
    It supports 2D and 3D mesh generation for general static and coupled temp-displacement analyses.
    
    Inputs:
        - model_name: Name of the model.
        - part_name: Name of the part.
        - size: Mesh size.
        - analysis_type: 'static' (default) or 'coupled_temp_disp' for coupled temperature-displacement analysis.
        - dimension: '2D' or '3D' (default) to specify the dimension of the model.
    
    Output:
        - The part is meshed with the specified size and element types based on the analysis type and dimension.
    '''
    p = mdb.models[model_name].parts[part_name]
    p.seedPart(size=size, deviationFactor=0.1, minSizeFactor=0.1)
    # Handle 3D elements
    if dimension == '3D':
        if analysis_type == 'static':
            elemType1 = mesh.ElemType(elemCode=C3D8, elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT)
            elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
            elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)
        elif analysis_type == 'coupled_temp_disp':
            elemType1 = mesh.ElemType(elemCode=C3D8T, elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT)
            elemType2 = mesh.ElemType(elemCode=C3D6T, elemLibrary=STANDARD)
            elemType3 = mesh.ElemType(elemCode=C3D4T, elemLibrary=STANDARD)
        
        pickedRegions = (p.cells,)
        p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, elemType3))
    # Handle 2D elements
    elif dimension == '2D':
        if analysis_type == 'static':
            elemType1 = mesh.ElemType(elemCode=CPS4, elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT)
            elemType2 = mesh.ElemType(elemCode=CPS3, elemLibrary=STANDARD)
        elif analysis_type == 'coupled_temp_disp':
            elemType1 = mesh.ElemType(elemCode=CPE4T, elemLibrary=STANDARD, secondOrderAccuracy=OFF, distortionControl=DEFAULT)
            elemType2 = mesh.ElemType(elemCode=CPE3T, elemLibrary=STANDARD)
        pickedRegions = (p.faces,)
        p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
    # Generate the mesh
    p.generateMesh()


def jobSubmit(model_name, JobName, numCpus=1, numGPUs=0, memory=90, userSubroutine='', precision=SINGLE):
    '''
    Submits and monitors a job for the specified model in Abaqus.
    
    Inputs:
        - model_name: Name of the model to run the job on.
        - JobName: Name of the job.
        - numCpus: Number of CPUs to use (default = 1).
        - numGPUs: Number of GPUs to use (default = 0).
        - memory: Percentage of memory to allocate (default = 90).
        - userSubroutine: Path to the user subroutine file, if any (default = '').
        - precision: Precision for the analysis, can be 'SINGLE' or 'DOUBLE' (default = 'SINGLE').
        
    Output:
        - The job is submitted, monitored, and results are extracted if successful.
    '''
    try:
        # Create and submit the job
        mdb.Job(name=JobName, model=model_name, description='', type=ANALYSIS, 
                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=memory, 
                memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
                explicitPrecision=precision, nodalOutputPrecision=precision, echoPrint=OFF, 
                modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine=userSubroutine, 
                scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=numCpus, 
                numGPUs=numGPUs)
        mdb.jobs[JobName].submit(consistencyChecking=OFF)
        mdb.jobs[JobName].waitForCompletion()
        if mdb.jobs[JobName].status == COMPLETED:
            print(JobName + " completed successfully.")
        else:
            print(JobName + " did not complete successfully. Status: {mdb.jobs[JobName].status}")
    except Exception as e:
        print("Failed to submit or complete job " + JobName + ": " + str(e))




# post

def get_node_label_by_coordinates_from_instance(model_name, instance_name, target_coords, tolerance=1e-3):
    '''
    This function returns the node label of the node closest to the given coordinates (x, y, z) in an instance.
    '''
    model = mdb.models[model_name]
    assembly = model.rootAssembly
    instance = assembly.instances[instance_name]
    closest_node = None
    min_distance = float('inf')
    for node in instance.nodes:
        node_coords = node.coordinates
        distance = ((node_coords[0] - target_coords[0])**2 + 
                    (node_coords[1] - target_coords[1])**2 + 
                    (node_coords[2] - target_coords[2])**2)**0.5
        if distance < tolerance and distance < min_distance:
            closest_node = node
            min_distance = distance
    if closest_node:
        return closest_node.label
    else:
        return None

def get_deformed_node_coordinates(odb, step_name, frame_number, instance_name, node_label):
    '''
    Cette fonction récupère les coordonnées déformées (x, y) d'un nœud après déformation dans un fichier .odb,
    en ignorant la composante z.
    '''
    instance = odb.rootAssembly.instances[instance_name]
    target_node = None
    for node in instance.nodes:
        if node.label == node_label:
            target_node = node
            break
    if target_node is None:
        odb.close()
        raise ValueError("Node label {} not found in instance {}".format(node_label, instance_name))
    
    step = odb.steps[step_name]
    frame = step.frames[frame_number]
    displacement_field = frame.fieldOutputs['U']
    disp = displacement_field.getSubset(region=instance).getSubset(region=target_node).values[0].data
    x_deformed = target_node.coordinates[0] + disp[0]
    y_deformed = target_node.coordinates[1] + disp[1]
    return (x_deformed, y_deformed)

def calculate_bending_angle(odb_path, step_name, frame_number, instance_name, node_label_1, node_label_2):
    '''
    Cette fonction calcule l'angle de flexion entre deux nœuds après déformation, dans l'intervalle [0°, 180°].
    '''
    odb = openOdb(odb_path)
    x1_deformed, y1_deformed = get_deformed_node_coordinates(odb, step_name, frame_number, instance_name, node_label_1)
    x2_deformed, y2_deformed = get_deformed_node_coordinates(odb, step_name, frame_number, instance_name, node_label_2)
    delta_x = x2_deformed - x1_deformed
    delta_y = y2_deformed - y1_deformed
    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)
    if angle_degrees <= 0:
        angle_degrees += 180
    return angle_degrees

def bending_angle_temp(odb_path, step_name, instance_name, node_label_1, node_label_2):
    '''
    Cette fonction calcule l'angle de flexion en fonction de la température pour chaque frame d'un step.
    '''
    odb = openOdb(odb_path)
    step = odb.steps[step_name]
    instance = odb.rootAssembly.instances[instance_name]
    result = []
    temperature = None
    tolerance = 0.01
    for frame_number, frame in enumerate(step.frames):
        angle = calculate_bending_angle(odb_path, step_name, frame_number, instance_name, node_label_1, node_label_2)
        if angle >= 180 - tolerance:
            continue
        temperature_field = frame.fieldOutputs['NT11']
        for value in temperature_field.values:
            if value.nodeLabel == node_label_1:
                    temperature = value.data
                    break
            if temperature is not None:
                result.append((temperature, angle))
            else:
                print("Température non trouvée pour le noeud")
    odb.close()
    p = os.getcwd()
    with open(p + '/result_data.txt', 'w') as f:
        for temp, angle in result:
            f.write("{},{}\n".format(temp, angle))
    print("Data saved to result_data.txt")
    return result


def find_frame_closest_temperature(odb_path, step_name, instance_name, y_target, target_temperature, tolerance=1e-5):
    '''
    Cette fonction retourne le numéro de la frame dont la température est la plus proche de la température cible.
    '''
    odb = openOdb(odb_path)
    step = odb.steps[step_name]
    instance = odb.rootAssembly.instances[instance_name]
    closest_frame = None
    min_temp_diff = float('inf') 
    for frame_number, frame in enumerate(step.frames):
        temperature = None
        temperature_field = frame.fieldOutputs['NT11']
        for node in instance.nodes:
            coordinates = node.coordinates
            if len(coordinates) == 3:
                _, y, _ = coordinates
            elif len(coordinates) == 2:
                _, y = coordinates 
            if abs(y - y_target) < tolerance:
                for value in temperature_field.values:
                    if value.nodeLabel == node.label:
                        temperature = value.data
                        break
                if temperature is not None:
                    temp_diff = abs(temperature - target_temperature)
                    if temp_diff < min_temp_diff:
                        min_temp_diff = temp_diff
                        closest_frame = frame_number
                break
    if closest_frame is not None:
        return closest_frame

def bending_angle_for_frame(odb_path, step_name, instance_name, node_label_1, node_label_2, frame_number, tolerance=0.01):
    '''
    Cette fonction calcule l'angle de flexion pour un numéro de frame spécifique et retourne l'angle et la température.
    '''
    odb = openOdb(odb_path)
    step = odb.steps[step_name]
    instance = odb.rootAssembly.instances[instance_name]
    temperature = None
    # Access the specific frame
    frame = step.frames[frame_number]
    angle = calculate_bending_angle(odb_path, step_name, frame_number, instance_name, node_label_1, node_label_2)
    if angle >= 180 - tolerance:
        print("L'angle de flexion "+str(angle)+"° est supérieur à la limite de tolérance pour la frame " + str(frame_number))
        odb.close()
        return None
    temperature_field = frame.fieldOutputs['NT11']
    for value in temperature_field.values:
        if value.nodeLabel == node_label_1:
            temperature = value.data
            break
    odb.close()
    if temperature is not None:
        return temperature, angle
    else:
        print("Température non trouvée pour le noeud.")
        return None



def bending_angle_temp_range(odb_path,file_name,model_name, step_name, instance_name, ptLayer,FinalTemp,initialTemp,pas,tolerance):
    node_label1 = get_node_label_by_coordinates_from_instance(model_name, instance_name, (ptLayer[1][0], 0, 0), tolerance)
    node_label2 = get_node_label_by_coordinates_from_instance(model_name, instance_name, (ptLayer[1][0] - 1e-3, 0, 0), tolerance)
    result = []
    for temp in range(initialTemp, FinalTemp, pas):
        frame_number = find_frame_closest_temperature(odb_path, step_name, instance_name.upper(), ptLayer[1][1], temp+273.15, tolerance)
        if frame_number is None:
            print("No frame found for temperature " + str(temp))
            continue
        angle_temp = bending_angle_for_frame(odb_path, step_name, instance_name.upper(), node_label1, node_label2, frame_number, tolerance)
        print(angle_temp)
        if angle_temp is not None:
            result.append(angle_temp)
    with open(file_name, 'w') as f:
        for x, y in result:
            f.write("{},{}\n".format(x, y))
    print("Data saved to {}".format(file_path))
    return result

def get_deformation_for_frame(odb_path, file_name,step_name, instance_name, y_target, n, tolerance=1e-5):
    '''
    Cette fonction retourne les déformations (en 2D: x, y) de tous les nœuds qui ont la même coordonnée y
    pour un numéro de frame spécifique.
    '''
    odb = openOdb(odb_path)
    step = odb.steps[step_name]
    instance = odb.rootAssembly.instances[instance_name]
    result = []
    frame_number = n
    frame = step.frames[frame_number]
    disp_x = []
    disp_y = []
    temperature = None
    displacement_field = frame.fieldOutputs['U']
    for node in instance.nodes:
        coordinates = node.coordinates
        if len(coordinates) == 3:
            x, y, _ = coordinates  # Ignore z if 3D
        elif len(coordinates) == 2:
            x, y = coordinates  # Only x and y if 2D
        if abs(y - y_target) < tolerance:
            for value in displacement_field.values:
                if value.nodeLabel == node.label:
                    displacement = value.data
                    if len(displacement) == 3:
                        u_x, u_y, _ = displacement  # Ignore z if 3D
                    elif len(displacement) == 2:
                        u_x, u_y = displacement  # Only u_x and u_y if 2D
                    disp_x.append(x + u_x)
                    disp_y.append(y + u_y)
                    break
    result = list(zip(disp_x, disp_y))
    file_path = 'deformation_data.txt'
    with open(file_name, 'w') as f:
        for x, y in result:
            f.write("{},{}\n".format(x, y))
    print("Data saved to {}".format(file_path))
    odb.close()
    return result


