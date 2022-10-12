# Import dependencies
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
import matplotlib.pylab as plt
import numpy as np
from pyvis.network import Network



#Section Tags: 
# Make List = mk_lst, Find Edges =  fnd_edg, Tree Growth = tr_grw, Graph Functions = gr_fnc, Streamlit = strm


#################################
####### M A K E   L I S T #######
############ mk_lst #############

# Takes a topology string and parses it into a list of individual topology elements
def Make_List(top_string, add_dna=True):  
    global elements 
    global linkers 
    global elements_list
    global linkers_list
    global top_list


    elements = 'ABCDEFGHIJKLMNOP'
    linkers = "abcdefgh"
    elements_list= list(elements)
    linkers_list= list(linkers)
    top_list=[]


    C1_search=True
    C2_search=False
    char_behind=None
    char_ahead=None
    two_char_behind=None
    two_char_ahead=None
    top_string=list(top_string)

    
    
    for idx, char in enumerate(top_string):

        #defines peek variables (characters before and after current idx value)
        if idx in range(1, len(top_string)):
            char_behind=top_string[idx-1]

        if idx in range(2, len(top_string)):
            two_char_behind=(top_string[idx-2])+(char_behind)

        if idx in range(0, len(top_string)-1):
            char_ahead=top_string[idx+1]

        if idx in range(0, len(top_string)-2):
            two_char_ahead=(char_ahead)+(top_string[idx+2])
        
        # For loop skips over syntactical topology characters ("!", "(", and ")")
        if char =='!':
            continue
        if char == '(' or char == ')':
            continue
        
        # The rest of the For loop identies elements based on leading and trailing syntactical characters
        #Identifies the start of a branched-cyclic element
        if C1_search is True and two_char_behind =='(!':
            C1_search=False
            C2_search=True
            top_list.append('(!' + char + ')')
            continue
        #identifies the start of a cyclic element 
        elif C1_search is True and char_behind == '!':
            C1_search=False
            C2_search=True
            top_list.append('!' + char)
            continue
        #identifies the end of a branched-cyclic element
        elif C2_search is True and two_char_ahead =='!)':
            C2_search=False
            top_list.append('(' + char + '!)')
            continue
        #identifies the end of a cyclic element 
        elif C2_search is True and char_ahead =='!':
            C2_search=False
            top_list.append(char + '!')
            continue
        #identifies a branching element 
        elif char_behind =='(' and char_ahead == ')':
            top_list.append('(' + char + ')')
            continue

        elif char in linkers or char in elements:
            top_list.append(char)
    if add_dna is True:
        top_list.append("DNA")
    return(top_list)




##################################
###### F I N D   E D G E S #######
############ fnd_edg #############

    #General Function Form: 
# Uses an outer For loop (loop A) that iteratively classifies elements of the list as either Regular, Cyclic, Branched, or Branched-Cyclic
# Uses an inner For loop (loop B) to re-iterate over the elements list and identifies edges based on specific criteria
# If criteria are met, adds edges to list of edges in the form [[A,B],...]

edges=[] #Edges stored in list as node pairs. List is appended by calling the Add_Edge Function.

def Add_Edge(E1, E2): #Adds edge for the two input elements (ie nodes)
    edges.append([E1,E2])


def Find_Edges(top_list): #Finds the edges for a list of ordered topology elements
    for idxA, A in enumerate(top_list): #Classifies A in outer For loop. 
        reg_A=False
        cyc_A1=False
        cyc_A2=False
        br_A=False
        br_cyc_A1=False
        br_cyc_A2=False

        
        if len(A)==1: #Flags Regular Elements
            reg_A=True
        if len(A)==2: #Flags starting Cyclic elements (ie !A)
            if A[0]=='!':
                cyc_start=A
                cyc_A1=True
            if A[1]=="!":
                cyc_A2=True
        if len(A)==3: #Flags Branched elements
            if A!='DNA':
                br_A=True
                br_point=top_list[idxA-1] # Defines the branch point as the element before the Branch element
        if len(A)==4: #Flags Branched-Cyclic elements.
            if A[1]=='!': #If Branched-Cyclic element is the start of a cycle
                br_cyc_start=A
                br_cyc_A1=True
                br_cyc_point=top_list[idxA-1] # Defines the branch point as the element before the Branched-Cyclic element
            if A[2]== '!': #If Branched-Cyclic element is the end of cycle
                br_cyc_A2=True
                br_cyc_end=A
                br_cyc_point=top_list[idxA-1] # Defines the branch point as the element before the Branched-Cyclic element
        
        #Classify B
        for idxB, B in enumerate(top_list):
            cyc_B=False
            br_cyc_B=False

            if A=='DNA' and B=='DNA':
                if '(' in top_list[(idxB-1)]:
                    Add_Edge(top_list[idxB-2],B)
                    break
                else:
                    Add_Edge(top_list[idxB-1],B)
                    break

            if A==B:
                continue

            if idxB == idxA+1: #If the B is the element following A
                if reg_A is True: #And if A is a regular element (not branched)
                    if top_list[idxA+1]=='DNA':#Continue on if the next A element is DNA (since DNA is handled with its own seperate rule above)
                        continue
                    else: #Add edge between consecutive elements A and B
                        Add_Edge(A,B)
                if cyc_A1 or cyc_A2 is True:
                    if top_list[idxA+1]=='DNA':#Continue on if the next A element is DNA (since DNA is handled with its own seperate rule above)
                        continue
                    else: #Add edge between consecutive elements !A and B
                        Add_Edge(A,B)

            if len(B)==2: #If B is a cyclic element
                if B[1]=='!': #If B is the end point of a cycle
                    cyc_end=B #Identify the the end point
                    cyc_B=True #flag the end point as found
                
            if len(B)==4: #If B is a branched cyclic element
                if B[2] =='!': #If B is the end point of a branched cycle
                    br_cyc_end=B #Identify the the end point
                    br_cyc_B=True #flag the end point as found

            if br_A is True: #Resolves the branching edge for a regular branching element
                if idxB == idxA+1:
                    Add_Edge(br_point,B)

            if br_cyc_A1 is True: #Resolves the branching edge for a starting branched cyclic element (the cyclic edge is resolved later down)
                if idxB == idxA+1:
                    Add_Edge(br_cyc_point,B)

            if br_cyc_A2 is True: #Resolves the branching edge for an ending branched cyclic element (the cyclic edge is resolved later down)
                if idxB == idxA+1:
                    if B!='DNA':
                        Add_Edge(br_cyc_point,B)
            
            if cyc_A1 is True: #Resolves cases involving regular cyclic starting points
                if cyc_B is True:
                    Add_Edge(cyc_start,cyc_end)
                if br_cyc_B is True:
                    Add_Edge(cyc_start,br_cyc_end)
            
            if br_cyc_A1 is True: #Resolves cases involving branched cyclic starting points
                if cyc_B is True:
                    Add_Edge(br_cyc_start,cyc_end)
                if br_cyc_B is True:
                    Add_Edge(br_cyc_start,br_cyc_end)
    return(edges)



###################################
###### T R E E   G R O W T H ######
############# tr_grw ##############

nodes=['A']
growth_control={'A':'inactive', 'A':'active'}
branches=[['A', 'AB']]
temp_cyclic_string=''

global cyclic_list
cyclic_list=[]


def Find_NodesAndEdges(A):
    for letter in A: #For each letter in the sequence list, consider all possible branched and cyclic permutations
        for current_node,value in list(growth_control.items()): #Iteratively goes through the growth control dictionary
            if value == 'active': #Looks for any active keys (ie nodes) in the growth control dict. 
                new_node=str(current_node)+letter #Create a new node by appending the existing active node with an unbranched version of the current letter from the list
                nodes.append(new_node)#Add the new node to the node list
                growth_control[new_node]='active'#Add the new node to the growth control dict as active
                growth_control[current_node]='inactive'#Sets the current growth control value to inactive
                if current_node=='A':
                    continue #Skips the branch addition step for the 'A' node since the (A, AB) branch is initialized in the branch list
                else:
                    branches.append([current_node,new_node]) #Add a branch (ie edge) between the current key (ie node) and the new_node
                if cycle_check == 'Yes': 
                    cyclic_tree_growth(current_node) #If cycle_check toggle is 'Yes' execute the cyclic_tree_growth algorithm
                                        #This algorithm adds nodes and branches for all acceptable cyclic permutations of the current key (ie node) 

                #if key[-1]==')':
                    #if cycle_check == 'Yes':
                        #cyclic_tree_growth()
                    #continue
                if '(' not in current_node:
                        new_br_node=str(current_node) + '(' + letter + ')'
                        nodes.append(new_br_node)
                        growth_control[new_br_node]='active'
                        branches.append([current_node,new_br_node])
                        #if cycle_check == 'Yes':
                            #cyclic_tree_growth()
                        if cycle_check == 'Yes':      
                            cyclic_tree_growth(new_br_node)
                            growth_control[new_br_node]='inactive'
        for current_node,value in list(growth_control.items()): #Iteratively goes through the growth control dictionary
            if value == 'active':
                if '!' not in current_node:
                    if cycle_check == 'Yes':
                        cyclic_tree_growth(current_node)
    return(nodes, branches)

#Finds all valid cyclic permutations of an acyclic topology list or string.
def Find_Cycles(list):

    global top_list
    global cyclic_list

    if isinstance(list,str):
        top_list=Make_List(list, add_dna=False)
    else:
        top_list=list

    cyclic_list=[]
    branch_span=False 
    #Outer for-loop (IdxA) controls all cyclic starting points
    for IdxA, element in enumerate(top_list): 
        temp_listA=top_list.copy() #Copies the toplist as a temp list
        inital_end_point=IdxA+2 #The minimum number of elements needed to complete a cycle is three, therefore the inital endpoint is started at IdxA+2
        if inital_end_point==len(top_list): #Checks to see if the initial endpoint had reached the end of the elements list
            break
        if len(top_list)<3:#Checks if the elements list contains a minimum of 3 terms
            break
        if '(' in element:#Checks if element is branched and if so, adds a cyclic-branched element in its place
            temp_listA[IdxA]="(!" + element[1] + ")"
        else:
            temp_listA[IdxA]="!" + element #Adds a cyclic element in place of current element
            start_point=IdxA
            for e in range(0, ((len(top_list))-1)):
                if '(' in top_list[start_point+1]:
                    branch_span=True
        #Inner for-loop (IdxB) controls all cyclic ending points
        for IdxB in range(inital_end_point, len(top_list)): #Considers the range between the inital cycle point and the end of elements list (all cycle endpoints necessarily occur in this range)
            temp_listB=temp_listA.copy()
            temp_cycle=''
            if branch_span is True:
                branch_span=False
                continue
            if '(' in top_list[IdxB]: # For handling branch points, ie (X)
                if branch_span is True:
                    continue
                for c in top_list[IdxB]:
                    if c == '(' or c == ')':
                        continue
                    else:
                        temp_listB[IdxB]="(" + c + "!)"
            else: # For handling regular elements
                temp_listB[IdxB]=top_list[IdxB] + "!" 
            for e in temp_listB:
                temp_cycle=temp_cycle + e
            cyclic_list.append(temp_cycle)
    return(cyclic_list)

def cyclic_tree_growth(node_variable):
    Find_Cycles(node_variable) #Takes the input acyclic key (ie node) and returns a list of possible cyclic topologies
    for seq in cyclic_list:
        nodes.append(seq) #Adds the cyclic string to the node list
        growth_control[seq]='inactive' #Adds the cyclic string to the growth control dict as inactive
        branches.append([node_variable,seq]) #Adds a branch between acyclic parent node and the cyclic child node


###########################################
###### G R A P H   F U N C T I O N S ######
################# gr_fnc ##################


# stores the vertices in the graph
vertices = []

# stores the number of vertices in the graph
vertices_no = 0
graph = []

global elements
global linkers

#Stores edge list values
edge_list=[]

def Construct_Graph(nodes, edges):


    node_colors=[]
    for element in nodes:
        if element == 'DNA':
            node_colors.append('#012A4A')
            continue
        for char in element:
            if char not in elements: 
                if char not in linkers and element != 'DNA':
                    continue
            else: 
                if 'A' in element:
                    node_colors.append('#A9D6E5')
                    break
                if 'B' in element:
                    node_colors.append('#61A5C2')
                    break
                if 'C' in element:
                    node_colors.append('#468FAF')
                    break
                if 'D' in element:
                    node_colors.append('#2A6F97')
                    break
                if 'E' in element:
                    node_colors.append('#014F86')
                    break
                if 'F' in element:
                    node_colors.append('#01497C')
                    break
                if 'G' in element:
                    node_colors.append('013A63')
                    break
                if 'H' in element:
                    node_colors.append('#013A63')
                    break
            if element in linkers and element != 'DNA': 
                node_colors.append('#FFE15C')


    node_shapes=[]
    for element in nodes:
        if element == 'DNA':
            node_shapes.append('dot')
        for char in element:
            if element=='DNA':
                continue
            if char in linkers:
                node_shapes.append('hexagon')
            if char in elements:
                node_shapes.append('dot')
            else:
                continue 
    return(node_colors, node_shapes)


def refresh_graph_info(): #Refreshes only graph related objects
    global edge_list
    global vertices
    global vertices_no
    global graph
    vertices=[]              
    vertices_no = 0
    graph = []
    edge_list=[]

def refresh_vars(): #Refreshes graph related objects and the current top_list
    global top_list
    global edge_list
    global vertices
    global vertices_no
    global graph
    top_list=[] 
    vertices=[]              
    vertices_no = 0
    graph = []
    edge_list=[]

### Add a vertex to the set of vertices and the graph
def add_vertex(v):
  global graph
  global vertices_no
  global vertices
  if v in vertices:
    #print("Vertex ", v, " already exists")
    pass
  else:
    vertices_no = vertices_no + 1
    vertices.append(v)
    if vertices_no > 1:
        for vertex in graph:
            vertex.append(0)
    temp = []
    for i in range(vertices_no):
        temp.append(0)
    graph.append(temp)

# Add an edge between vertex v1 and v2 with edge weight e
def add_edge(v1, v2, e):
    global graph
    global vertices_no
    global vertices
    if v1 not in vertices:
        pass
        #print("Vertex ", v1, " does not exist.")
    elif v2 not in vertices:
        pass
        #print("Vertex ", v2, " does not exist.")
    else:
        index1 = vertices.index(v1)
        index2 = vertices.index(v2)
        graph[index1][index2] = e

# Print the graph
def print_graph():
  global graph
  global vertices_no
  for i in range(vertices_no):
    for j in range(vertices_no):
      if graph[i][j] != 0:
        print(vertices[i], " -> ", vertices[j], \
        "reaction ID: ", graph[i][j])

# make edge list
def make_edge_list():
  global graph
  global vertices_no
  for i in range(vertices_no):
    for j in range(vertices_no):
      if graph[i][j] != 0:
        X=vertices[i]
        Y=vertices[j]
        edge_list.append([X,Y])
    return(edge_list)


###############################
###### S T R E A M L I T ######
############# strm ############

st. set_page_config(layout="wide")

# Set header title
st.title('DEL Topology Visualization')

col1, col2 = st.columns([2, 1], gap="small")

#Define number of DEL elements from streamlit user input
DEL_size= int(st.sidebar.selectbox('Number of diversity elements:',('3', '4', '5', '6', '7', '8')))

# Toggle consideration of cyclic topologies from streamlit 
cycle_check = st.sidebar.radio('Consider cyclic topologies?',('Yes', 'No'), index=1)

# Initialization value for scaling node size by literature precedence
lit_scale='No'

# Option for scaling node size by literature precedence appears if cycle check is "Yes"
if cycle_check == 'Yes':
    lit_scale= st.sidebar.radio('Scale by literature prevalance?',('Yes', 'No'), index=1)

# Allows user to switch between Dendridic or Hierarchical tree layout.
view= st.sidebar.radio ('Tree Layout:', ('Dendridic', 'Hierarchical'))

DE_string='ABCDEFGHIJK'
DE_selection=DE_string[:DEL_size]

sequence=DE_selection[1:]
seq_list=list(sequence)

#teststring='A!BCD(E!)F'


nodes, edges=Find_NodesAndEdges(seq_list)
print(nodes)
print(edges)

if lit_scale == 'Yes':
    df=pd.read_csv('Descriptor_Data.csv')

    temp_dict=df.set_index('Entry').to_dict()
    for k,v in temp_dict.items():
        occurance_dict=v



TREE= Network(height='800px',width='1000px')

#TREE.add_nodes(nodes)

if lit_scale == 'No':
    for node in nodes:
        if '!' in node:
            TREE.add_node(node, shape="square", color="#fff9ae")
        else:
            TREE.add_node(node, shape="dot")

if lit_scale == 'Yes':
    for node in nodes:
        if node in occurance_dict.keys():
            if '!' in node:
                TREE.add_node(node, shape="square", color="#c6a8e0", size=occurance_dict.get(node)*10)
            else:
                TREE.add_node(node, shape="dot", color="#c6a8e0", size=occurance_dict.get(node)*10)
        elif '!' in node:
                TREE.add_node(node, shape="square", color="#fff9ae", size=10)
        else:
            TREE.add_node(node, shape="dot",size=10)

TREE.add_edges(edges)

#Uncomment this to generate the HTML file with buttons to adjust viewing options
#TREE.show_buttons()


if view=='Hierarchical':
    TREE.set_options("""
    const options = {
    "layout": {
        "hierarchical": {
        "enabled": true,
        "levelSeparation": 300,
        "nodeSpacing": 85,
        "treeSpacing": 150
        }
    },
    "physics": {
        "hierarchicalRepulsion": {
        "centralGravity": 0,
        "avoidOverlap": null
        },
        "minVelocity": 0.75,
        "solver": "hierarchicalRepulsion"
    }
    }
    """)


#TREE.show( 'Tree' + ".html")


# Save and read graph as HTML file (on Streamlit Sharing)
try:
   path = '/tmp'
   TREE.save_graph(f'{path}/Tree.html')
   HtmlFile1 = open(f'{path}/Tree.html','r',encoding='utf-8')
# Save and read graph as HTML file (locally)
except:
    path = './'
    TREE.save_graph(f'{path}/Tree.html')
    HtmlFile1 = open(f'{path}/Tree.html','r',encoding='utf-8')


edges=[]
node_selection= st.sidebar.selectbox('Choose a node to inspect:',(nodes))


top_nodes=Make_List(str(node_selection),add_dna=True)
top_edges=Find_Edges(top_nodes)
colors, shapes= Construct_Graph(top_nodes,top_edges)

DEL= Network()

DEL.add_nodes(top_nodes, color=colors, shape=shapes)

DEL.add_edges(top_edges)

DEL.show("del.html")

# Save and read graph as HTML file (on Streamlit Sharing)
try:
    path = '/tmp'
    DEL.save_graph(f'{path}/del.html')
    HtmlFile2 = open(f'{path}/del.html','r',encoding='utf-8')
# Save and read graph as HTML file (locally)
except:
    path = './'
    DEL.save_graph(f'{path}/del.html')
    HtmlFile2 = open(f'{path}/del.html','r',encoding='utf-8')
with col2:
    st.header("Topology Explorer")
    with st.container():
        components.html(HtmlFile2.read(), height=800)
with col1:
    st.header("Topology Tree")
    with st.container():
        components.html(HtmlFile1.read(), height=805)
