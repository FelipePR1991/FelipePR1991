# Steam Machine Learning Test 07

import os
import requests
import pandas as pd
import wmi
from sklearn.neighbors import NearestNeighbors
import numpy as np
from cryptography.fernet import Fernet
import json

# Função para carregar e descriptografar as variáveis sensíveis
def load_sensitive_data():
    key_file_path = r'C:\Users\USER\Desktop\Python\key.txt'

    with open(key_file_path, 'r') as key_file:
        key = key_file.read().strip()

    cipher_suite = Fernet(key.encode())

    enc_file_path = r'C:\Users\USER\Desktop\Python\config.enc'

    with open(enc_file_path, 'rb') as file:
        encrypted_data = file.read()

    decrypted_data = cipher_suite.decrypt(encrypted_data)

    config_data = json.loads(decrypted_data.decode())

    return config_data.get('STEAM_ID'), config_data.get('API_KEY')

# Função para capturar informações do sistema
def get_system_info():
    w = wmi.WMI()
    
    # Obter informações de CPU
    cpu_name = None
    cpu_cores = 0
    cpu_threads = 0
    for cpu in w.Win32_Processor():
        cpu_name = cpu.Name
        cpu_cores = cpu.NumberOfCores
        cpu_threads = cpu.NumberOfLogicalProcessors

    # Obter informações de GPU
    gpu_name = None
    gpu_memory = 0
    for gpu in w.Win32_VideoController():
        gpu_name = gpu.Name
        gpu_memory = gpu.AdapterRAM
        if gpu_memory is None or gpu_memory <= 0:
            gpu_memory = 2 * 1024**3  # Assumindo que a GT 740 tem 2GB de VRAM

    # Obter informações de RAM
    total_ram = 0
    for mem in w.Win32_ComputerSystem():
        total_ram = int(mem.TotalPhysicalMemory) / (1024**3)  # Em GB

    # Retornar as informações coletadas
    return {
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
        "cpu_threads": cpu_threads,
        "gpu_name": gpu_name,
        "gpu_memory": gpu_memory / (1024**3),  # Em GB
        "total_ram": total_ram
    }

# Função para filtrar jogos compatíveis com o hardware
def filter_games_based_on_hardware(possible_games_df, system_info):
    possible_games_df['min_cpu_cores'] = [
        1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 
        1, 1, 1, 2, 1, 1, 2, 2, 1, 1,
        2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  
        4, 4  
    ]
    possible_games_df['min_gpu_memory'] = [
        0.032, 0.128, 0.128, 0.128, 0.256, 0.128, 0.128, 0.064, 0.064, 0.064, 
        0.128, 0.064, 0.032, 0.128, 0.064, 0.128, 0.128, 0.128, 0.064, 0.064, 
        0.128, 0.128, 0.128, 0.032, 0.064, 0.128, 0.256, 0.256, 0.128, 0.064, 
        0.064, 0.064, 0.064, 0.064, 0.064, 0.032, 0.032, 0.032, 0.064, 0.064, 
        0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032,  
        2, 2  
    ]  # em GB
    possible_games_df['min_ram'] = [
        0.256, 0.512, 2, 0.512, 2, 2.5, 1, 1, 1, 1, 
        1, 0.512, 0.256, 2, 0.512, 1, 2, 2, 0.512, 0.512, 
        2, 0.5, 0.5, 0.512, 0.512, 1, 2, 2, 2, 0.256, 
        0.256, 0.256, 0.256, 0.512, 0.256, 0.256, 0.256, 0.512, 0.256, 0.256, 
        0.256, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032,  
        8, 8  
    ]

    # Filtrar jogos compatíveis com o hardware do sistema
    compatible_games_df = possible_games_df[
        (possible_games_df['min_cpu_cores'] <= system_info['cpu_cores']) &
        (possible_games_df['min_gpu_memory'] <= system_info['gpu_memory']) &
        (possible_games_df['min_ram'] <= system_info['total_ram'])
    ]

    return compatible_games_df

# Captura de dados dos jogos do Steam
def get_steam_games(api_key, steam_id):
    url = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={api_key}&steamid={steam_id}&format=json&include_appinfo=true"
    response = requests.get(url)

    try:
        games = response.json().get('response', {}).get('games', [])
        print(f"Número de jogos encontrados: {len(games)}")

        games_df = pd.DataFrame(games)

        if 'name' in games_df.columns and 'playtime_forever' in games_df.columns:
            games_df['hours_played'] = games_df['playtime_forever'] / 60
            games_df = games_df[['name', 'hours_played']].sort_values(by='name')
            return games_df
        else:
            print("As colunas 'name' ou 'playtime_forever' não foram encontradas.")
            return pd.DataFrame()
    except Exception as e:
        print("Erro ao processar a resposta JSON:", e)
        return pd.DataFrame()

# Função para criar o modelo de aprendizado de máquina e recomendar jogos
def machine_learning_recommendation(games_df, possible_games_df, num_recommendations=30):
    owned_games = set(games_df['name'].str.lower())

    possible_games_df = possible_games_df[~possible_games_df['name'].str.lower().isin(owned_games)]

    if possible_games_df.empty:
        print("Nenhum jogo novo disponível para recomendar.")
        return []

    X_train = np.array(games_df['hours_played']).reshape(-1, 1)

    knn = NearestNeighbors(n_neighbors=num_recommendations, algorithm='auto')
    knn.fit(X_train)

    X_test = np.zeros((possible_games_df.shape[0], 1))
    distances, indices = knn.kneighbors(X_test)

    possible_games_df['distance'] = distances.mean(axis=1)
    recommended_games = possible_games_df.sort_values(by='distance', ascending=True)['name'].head(num_recommendations).tolist()

    return recommended_games

# Capturar as informações do sistema
system_info = get_system_info()
print("Informações do Sistema:", system_info)

# Carregar e descriptografar as variáveis sensíveis
steam_id, api_key = load_sensitive_data()

# Obter dados dos jogos do Steam
games_df = get_steam_games(api_key, steam_id)

possible_games = [
    "Half-Life 2", "Portal", "Left 4 Dead 2", "Team Fortress 2", "Stardew Valley",
    "Terraria", "Don't Starve", "FTL: Faster Than Light", "Binding of Isaac", 
    "Undertale", "Celeste", "Braid", "The Elder Scrolls III: Morrowind",
    "Factorio", "Grim Fandango", "Psychonauts", "Hollow Knight", 
    "Hyper Light Drifter", "Risk of Rain", "Fez", "Hotline Miami", 
    "Mark of the Ninja", "Shovel Knight", "Slay the Spire", "Papers, Please", 
    "Spelunky", "Darkest Dungeon", "Dead Cells", "Cuphead", "Oxenfree",
    "Heavy Gear 2", "Eden", "Unreal 1999", "Outpost", "Urban Chaos",
    "Deus Ex", "System Shock 2", "Quake", "Unreal Tournament", 
    "Max Payne", "Thief: The Dark Project", "Baldur's Gate", 
    "Planescape: Torment", "Fallout 2", "Star Wars: Knights of the Old Republic", 
    "Diablo II", "StarCraft", "Warcraft III", "Age of Empires II", 
    "Command & Conquer: Red Alert", "The Secret of Monkey Island",
    "DOOM (2016)"
]

possible_games_df = pd.DataFrame(possible_games, columns=['name'])

possible_games_df = filter_games_based_on_hardware(possible_games_df, system_info)

if not games_df.empty and not possible_games_df.empty:
    suggested_games = machine_learning_recommendation(games_df, possible_games_df, num_recommendations=30)
    
    print("Sugestões de jogos:")
    for idx, game in enumerate(suggested_games, start=1):
        print(f"({idx}) {game}")
else:
    print("Nenhum jogo foi encontrado para análise ou nenhum jogo é compatível com o seu sistema.")

