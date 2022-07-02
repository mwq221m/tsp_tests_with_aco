import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Aco():
    def __init__(self,size,iteration,city_num,alpha,beta,rho,Q,distance_matrix):
        self.size=size
        self.iteration=iteration
        self.city_num=city_num
        self.fitness=np.zeros(self.size)
        #self.distance_matrix=np.zeros((self.city_num,self.city_num))
        self.distance_matrix=distance_matrix
        for i in range(self.city_num):
            self.distance_matrix[i,i]=1e-3
        self.eta_matrix=1/self.distance_matrix
        self.pheromone_matrix=np.ones((self.city_num,self.city_num))
        for i in range(self.city_num):
            self.pheromone_matrix[i][i]=0
        self.delta_pheromone_matrix=np.zeros((self.city_num,self.city_num))
        #self.population=np.random.randint(0,size=(self.size,self.city_num))
        temp=np.random.randint(self.city_num,size=self.size)
        self.population=[]
        for i in range(self.size):
            self.population.append([temp[i]])
        self.alpha=alpha
        self.beta=beta
        self.rho=rho
        self.Q=Q
        self.banned_list=[]
        for i in range(self.size):
            self.banned_list.append([])

    def run_one_step(self):
        for i in range(self.size):
            individual=self.population[i]
            present_point=individual[-1]
            p_temp=np.zeros(self.city_num)
            sum_temp=0
            for j in range(self.city_num):
                if j not in individual and j!=present_point:
                    sum_temp+=(self.pheromone_matrix[present_point,j])**self.alpha*(self.eta_matrix[present_point,j])**self.beta
            for j in range(self.city_num):
                if j not in individual and j!=present_point:
                    p_temp[j]=(self.pheromone_matrix[present_point,j])**self.alpha*(self.eta_matrix[present_point,j])**self.beta
            p_temp=p_temp/sum_temp
            choice_temp=np.random.choice(a=np.arange(self.city_num),size=1,p=p_temp)
            choice_temp=int(choice_temp)
            individual.append(choice_temp)

    def run_one_loop(self):
        for i in range(self.city_num-1):
            self.run_one_step()

    def fitness_calculation(self):
        for i in range(self.size):
            sum_temp=0
            individual=self.population[i]
            for j in range(self.city_num-1):
                start=individual[j]
                end=individual[j+1]
                sum_temp+=self.distance_matrix[start,end]
            sum_temp+=self.distance_matrix[individual[-1],individual[0]]
            self.fitness[i]=sum_temp

    def update_pheromone(self):
        self.pheromone_matrix *= 1 - self.rho
        for i in range(self.size):
            individual=self.population[i]
            loop_temp=0
            for j in range(len(individual)-1):
                start=individual[j]
                end=individual[j+1]
                loop_temp+=self.distance_matrix[start,end]
            det_temp=self.Q/loop_temp
            #self.pheromone_matrix*=1-self.rho
            for j in range(len(individual)-1):
                start=individual[j]
                end=individual[j+1]
                #self.pheromone_matrix[start,end]=(1-self.rho)*self.pheromone_matrix[start,end]
                self.pheromone_matrix[start,end]+=det_temp

    def reset_population(self):
        temp = np.random.randint(self.city_num, size=self.size)
        self.population = []
        for i in range(self.size):
            self.population.append([temp[i]])






#test=Aco(size=50,iteration=500,city_num=30,alpha=4,beta=3,rho=0.2,Q=2)
data=pd.read_excel('data_generation.xlsx')
x=data['x']
y=data['y']
#plt.figure()
#plt.scatter(x,y)
#plt.show()
distance_matrix=np.zeros((len(x),len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        distance_matrix[i, j] = ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) ** 0.5
#test.distance_matrix=distance_matrix
test=Aco(size=50,iteration=100,city_num=30,alpha=4,beta=3,rho=0.2,Q=2,distance_matrix=distance_matrix)
best_list=[]
min_temp=1e4
for i in range(test.iteration):
    test.run_one_loop()
    test.fitness_calculation()
    if test.fitness.min()<min_temp:
        min_temp=test.fitness.min()
        idx=np.where(test.fitness==test.fitness.min())[0][0]
        best_individual=test.population[idx]
    best_list.append(min_temp)

    test.update_pheromone()
    if i==test.iteration-1:
        continue
    test.reset_population()
plt.figure()
for i in range(len(best_individual)-1):
    start=best_individual[i]
    end=best_individual[i+1]
    plt.scatter([x[start],x[end]],[y[start],y[end]])
    plt.plot([x[start], x[end]], [y[start], y[end]])
first=best_individual[0];last=best_individual[-1]
plt.scatter([x[last],x[first]],[y[last],y[first]])
plt.plot([x[last],x[first]],[y[last],y[first]])
plt.show()

plt.figure()
plt.plot(best_list)
plt.show()
print('最佳路径',best_individual)
print('最短距离',min_temp)






