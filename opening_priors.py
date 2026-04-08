import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("LICHESS_TOKEN")

def get_games(moves: str):
    r = requests.get(f"https://explorer.lichess.org/lichess?{moves}",
                     headers={"Authorization": f"Bearer {TOKEN}"})
    
    if r.status_code != 200:
        print(f"Check authentication")  
    
    data = r.json()
    return data["white"] + data["black"] + data["draws"] 


print(get_games(""))