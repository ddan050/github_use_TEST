import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from collections import namedtuple
from scipy.stats import rankdata
import pandas as pd
import io


Transition = namedtuple('Transition', ('state', 'action', 'cost', 'next_state'))

def personalize_history_version2(init_state,store_state,reward,iteration):
    init_state = np.array(init_state)
    store_state = np.array(store_state)
    alpha = 0.1
    init_state = init_state + alpha * reward * (np.asarray(store_state))
    init_state = init_state / (iteration+1)

    return init_state

'''def personalize_update(store_info, prefer):
    store_info = np.asarray(store_info)
    prefer = np.asarray(prefer)
    person_prefer = np.linalg.solve(np.dot(store_info.T, np.dot(np.diag(prefer), store_info)) + 0.01 * np.eye(10),
                         np.dot(store_info.T, np.dot(np.diag(prefer), prefer.T))).T
    return person_prefer'''


class DQN(nn.Module):
    def __init__(self, base_state, store_info):
        super(DQN, self).__init__()
        self.x_dim = 15 # item 별 정보로 state 구성
        self.y_dim = 76  # 각 item의 value 계산

        self.input_linear = nn.Linear(self.x_dim, 100)  # policy 구성 조건에 따라 input dimension 달라짐
        self.output_linear = nn.Linear(100, self.y_dim)  # output dimension 달라짐 (CI:matrix 형태, CD:item별 value)
        self.fn1 = nn.Linear(100, 100) # 2개의 hidden layer와 100개의 unit
        self.fn2 = nn.Linear(100, 100)

        self.buffer_limit = 300
        self.replay_buffer = []
        self.position = 0
        self.base_state = np.array(base_state)
        self.store_information = store_info

    def forward(self, x):
        x= torch.reshape(x,[-1, self.x_dim])
        x = self.input_linear(x)
        x = F.relu(self.fn1(x))
        x = F.relu(self.fn2(x))
        q_pred = self.output_linear(x)
        return q_pred

    def scheduling(self, state):
        q_value = main_net(state)
        q_value = q_value.detach().numpy()

        return q_value

    def click_event(self, person_prefer, ranking_store, epsilon):  # 시나리오 관련 부분, click_ok 부분 구현 필요
        score_list = np.zeros(6)  # 그럼 선택을 아예 안할 때에 대한 기준 필요
        person_prefer = np.array(person_prefer)
        store_info = np.array(self.store_information)
        for i in range(6):
            store = store_info[int(ranking_store[i])][:]
            score_list[i] = np.dot(person_prefer, store)
        check = np.where(score_list <= 0.5)
        rnd = np.random.rand(1)
        if rnd < epsilon:
            click_index = np.random.randint(6) - 1
        else:
            if len(check) == 6:
                click_index = 8
            else:
                click_index = np.argmax(score_list)
        return click_index

    def personalize_history(self, init_state, reward):  # 수정 필요 :: 그냥 algorithm 참고해서 하면 안됨! 사용자 기반 base 취향 쓰지 말기
        alpha = 0.1
        discount = 0.9
        init_state = np.array(init_state)
        init_state = init_state - 2 * alpha * (
                    (np.transpose(init_state).dot(self.base_state) - reward) * self.base_state + discount * init_state)
        self.base_state = self.base_state - 2 * alpha * (
                    (np.transpose(self.base_state).dot(init_state) - reward) * init_state + discount * self.base_state)
        return init_state, self.base_state

    def train(self):  # == 교수님 코드에서 agent_dqn : optimize_model
        batch_size = 30  # supp 문서보고 숫자 수정 필요
        discount = 0.9  # 수정 필요
        transition = main_net.sample(batch_size)
        batch = Transition(*zip(*transition))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        cost_batch = torch.cat(batch.cost)
        next_state_batch = torch.cat(batch.next_state)

        state_action_values = main_net(state_batch).gather(1, action_batch.type(torch.int64))

        next_state_values = target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values.reshape(batch_size, 1)) * discount + cost_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values)
            # print("loss=", loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def put(self, *args):
        self.replay_buffer.append(Transition(*args))
        self.position = (self.position+1) % (self.buffer_limit)

    def sample(self, batch_size):
        mini_batch = random.sample(self.replay_buffer, batch_size)
        return mini_batch

def making_base_prefer(person_prefer, store_info):
    person_prefer = np.asarray(person_prefer)
    person_prefer_check = person_prefer[0:10]
    store_info_check = np.asarray(store_info)
    store_info_check = store_info_check[0:76, 0:10]
    base_prefer = np.dot(store_info_check, (person_prefer_check.T))
    store_index = np.argmax(base_prefer)
    target_prefer = person_prefer + 0.01 * (np.asarray(store_info[store_index]))

    return target_prefer


optimal_person_prefer = [0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] # 몇 가지의 정보만 안다고 가정하고 설정
person_prefer = [0.1, 0.6, 0.3, 0.1, 0.2, 0.4, 0.1, 0.1, 0.4, 0.2, 0.5, 0.1, 0.1, 0.2, 0.1]
store_info = pd.read_csv('C:/Users/82104/Downloads/store_info.csv')
review_data = pd.read_csv('C:/Users/82104/Downloads/naver_review_data.csv')
first_store_info = store_info.loc[0:77, ["type", "vegan", "meat", "seafood", "raw", "spice"]]
review_data_info = review_data.loc[0:77, ["taste", "special", "interior", "kind", "wide", "total"]]
first_store_info = first_store_info.drop(37, axis=0)
review_data_info = review_data_info.drop(37, axis=0)
first_store_info = pd.get_dummies(first_store_info, columns=['type'])
review_data_info["taste"] = review_data_info["taste"].astype(float)
review_data_info["special"] = review_data_info["special"].astype(float)
review_data_info["interior"] = review_data_info["interior"].astype(float)
review_data_info["kind"] = review_data_info["kind"].astype(float)
review_data_info["wide"] = review_data_info["wide"].astype(float)
review_data_info["total"] = review_data_info["total"].astype(float)
review_data_info["taste"] = review_data_info["taste"]/review_data_info["total"]
review_data_info['special'] = review_data_info['special']/review_data_info['total']
review_data_info['interior'] = review_data_info['interior']/review_data_info['total']
review_data_info['kind'] = review_data_info['kind']/review_data_info['total']
review_data_info['wide'] = review_data_info['wide']/review_data_info['total']
review_data_info = review_data_info.drop(['total'],axis=1)
first_store_info = pd.concat([first_store_info, review_data_info],axis=1)
first_store_info = first_store_info.values.tolist()

total_reward = np.zeros(50000)
gamma = 0.9
interval = 300
lr = 0.001
size = 0
iteration = 0
ranking_store = np.zeros(6)
click_reward = np.zeros(6)
target_prefer = making_base_prefer(person_prefer, first_store_info)
location_event = 0
main_net = DQN(target_prefer, first_store_info)
target_net = DQN(target_prefer, first_store_info)
target_net.load_state_dict(main_net.state_dict())
optimizer = optim.Adam(main_net.parameters(), lr=0.001)

for iteration in range(50000):
    if iteration > 10000:
        epsilon = 1 / iteration
    else:
        epsilon = 0.1
    cost = 0
    person_prefer = copy.deepcopy(person_prefer)
    state = torch.FloatTensor(person_prefer)
    prefer_score = main_net.scheduling(state)
    prefer_score = prefer_score.reshape(-1)
    candidate_list = prefer_score

    #location_event = input("위치를 입력하세요: ")

    if iteration > 25000:
        if iteration % 4 == 0:
            location_event = '0'
        elif iteration % 4 == 1:
            location_event = '1'
        elif iteration % 4 == 2:
            location_event = '2'
        elif iteration % 4 == 3:
            location_event = '3'

    if location_event == '1' : #건대
        candidate_list = candidate_list[0:24]
    elif location_event == '2' : #성수
        candidate_list = candidate_list[24:48]
    elif location_event == '3' : #강남
        candidate_list = candidate_list[48:77]

    ranks = rankdata((-candidate_list), method='min')  # 순위를 q_value 기준 내림차순으로 해야하므로 (-1) 처리
    for j in range(1, 7):
        for i in range(len(candidate_list)):
            if ranks[i] == j:
                if location_event == '2' :
                    ranking_store[j - 1] = (i+24)
                elif location_event == '3' :
                    ranking_store[j - 1] = (i+48)
                else :
                    ranking_store[j - 1] = i

    ran = np.random.rand(1)
    if ran < epsilon: #random하게 선택할 때 중복 고려해야함 -> 아이디어..?
        ranking_store[4] = np.random.randint(76)
        ranking_store[5] = np.random.randint(76)
    print('ranking=', ranking_store)
    ##### list 안에 가게 index 저장 => 그래야 app에 보낼 수 있다 ########
    ##### click_store = receive_feedback() -> 내가 준 list 내에서 선택된 가게 index를 받아옴 ######
    if iteration % 500 == 0 :
        print('ranking=', ranking_store)
        print_store_info = np.asarray(first_store_info)
        #print('store_info=',print_store_info[int(ranking_store[0])])
    click_store = main_net.click_event(person_prefer, ranking_store, epsilon)
    click_reward = [0, 0, 0, 0, 0, 0]
    if click_store != 8 :
        click_reward[click_store] = 1
        update_person_prefer, update_target_prefer = main_net.personalize_history(person_prefer,
                                                                                  click_reward[click_store])
    else :
        click_reward = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
        for i in range(6):
            update_person_prefer, update_target_prefer = main_net.personalize_history(person_prefer,
                                                                                      click_reward[i])
     # click한 가게랑 1위 가게랑 일치하면 +1 / 아니면 0 / 아무 것도 선택하지 않았다면 -0.5
    #update_person_prefer = personalize_history_version2(person_prefer, first_store_info[int(ranking_store[click_store])], click_reward[click_store],iteration)
    person_prefer = update_person_prefer.tolist()
    if iteration % 500 == 0 :
        print("update prefer=",person_prefer)
    next_state = torch.FloatTensor(person_prefer)
    for i in range(6):
        main_net.put(state, torch.tensor(ranking_store[i]).reshape(-1, 1).int(), torch.tensor(click_reward[i]).reshape(-1, 1).float(), next_state)
        size += 1

    if size >= 300:
        main_net.train()

    if (iteration % interval == 0) and (iteration != 0):
        target_net.load_state_dict(main_net.state_dict())

torch.save(DQN, 'capston_model_main.pt')


model_script = torch.jit.script(main_net)
model_script.save('capston_model_script2.pt')



'''import matplotlib.pyplot as plt
print("user preference=", update_person_prefer)
show_reward = np.zeros(50000)
total = 0
dqn_total = 0
time = 1
for i in range(50000):
    total += total_reward[i]
    show_reward[i] = total / time
    time += 1
plt.plot(show_reward)
plt.show()'''
