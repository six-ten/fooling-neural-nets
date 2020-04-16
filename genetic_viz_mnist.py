import streamlit as st
from tensorflow.keras import models
import tensorflow as tf 
import cv2 
import numpy as np
import time 
from deap import creator , base, tools 
import random
import matplotlib.pyplot as plt 
with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

header_image = cv2.imread('./header.jpg')
st.image(header_image,use_column_width=True)

debug = st.empty()

model = None
model2 = None
#@st.cache(persist=True)
def load_model():
    global model,model2 
    model = tf.keras.models.load_model('MNIST.h5')
    model2 = tf.keras.models.load_model('MNIST986.hs')

load_model()

st.markdown("***pick a number to increase \n prediction confidence for***")
#selection = st.selectbox("Select a Number ",tuple([i for i in range(10)]))
selection = st.number_input("select a number",0,9,value=0)
img = st.empty()
st.markdown('**PREDICTION CONFIDENCE**')
chart = st.empty()
pred = st.empty()
chart2 = st.empty()
progress_bar = st.progress(0)
#selection = st.selectbox("**pick a number to increase \n prediction confidence for",tuple([i for i in range(10)]))

fitness_weights = tuple([ 1 if x == selection else 0 for x in range(10)])
info = st.empty()
def FITNESS(individual):
    individual = list(individual)
    
    x = np.array(individual)
    x = np.array(x,dtype='uint8')
    f2 = -(sum(x)/np.prod(x.shape))*(100/255)
    x = x.reshape((1,28,28,1))
    x = x / 127.5 -1 
    #debug.text(x)
    prediction = model.predict(x)[0]
    fitness = prediction*100*(np.array(fitness_weights))
    #debug.text(str(prediction)+str(fitness))
    return sum(fitness)+f2*0.5,

def PREDICT(model,I):
    x = I
    
    x = np.reshape(x,(1,28,28,1))
    x = x / 127.5 -1 
    #debug.text(x)
    prediction = model.predict(x)[0]
    
    return prediction

# GA code -------------------------



VECTOR_LENGTH = 28*28

POPULATION_SIZE = st.sidebar.slider("population size :",25,150,100,step=10)
P_CROSSOVER = st.sidebar.slider("crossover probability :",0.4,1.0,0.9,step = 0.1)
P_MUTATION = st.sidebar.slider("mutation probability :",0.2,1.0,0.4,step = 0.1)

MAX_GENERATIONS = st.sidebar.slider("Maximum Generations:",60,2000,1000)

RANDOM_SEED = 43 
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()
toolbox.register("random255",random.randint,0,255)

creator.create("FitnessMax",base.Fitness, weights=(1.0,))
creator.create("Individual",list,fitness=creator.FitnessMax)

toolbox.register("individualCreator",tools.initRepeat,
    creator.Individual,toolbox.random255,VECTOR_LENGTH
    )
#st.help(toolbox.register)

toolbox.register("evaluate", FITNESS)
toolbox.register("populationCreator",tools.initRepeat,list,
    toolbox.individualCreator)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit,indpb=1.0/VECTOR_LENGTH)
#st.help(toolbox.populationCreator)
population = toolbox.populationCreator(n=POPULATION_SIZE)
generationCounter = 0

fitnessValues = list(map(toolbox.evaluate, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

fitnessValues = [individual.fitness.values[0] for individual in population]

maxFitnessValues = []
meanFitnessValues = []


# GA code -------------------------


# main loop



while generationCounter < MAX_GENERATIONS :
    generationCounter += 1
    progress_bar.progress(int((generationCounter/MAX_GENERATIONS)*100))
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    for mutant in offspring:
        if random.random() < P_MUTATION:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
    freshFitnessValues = list(map(toolbox.evaluate,freshIndividuals))
    for individual, fitnessValue in zip(freshIndividuals,freshFitnessValues):
        individual.fitness.values = fitnessValue
        #debug.text(str(fitnessValue))
    population[:] = offspring
    fitnessValues = [ind.fitness.values[0] for ind in population]
    maxFitness = max(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)
    info.text("- Generation {}: Max Fitness = {:.2f}, Avg Fitness = {:.2f}"
    .format(generationCounter, maxFitness, meanFitness))
    best_index = fitnessValues.index(max(fitnessValues))
    best_individual = population[best_index]
    I = np.array(best_individual,dtype='uint8')
    I = np.reshape(I,(28,28,1))
    if generationCounter % 5 == 0 or generationCounter == MAX_GENERATIONS:
        img.image(I,width=300,caption='Generation :'+ str(generationCounter))
    chart.bar_chart(PREDICT(model,I))
    chart2.bar_chart(PREDICT(model2,I))
    #chart.line_chart(PREDICT(I))