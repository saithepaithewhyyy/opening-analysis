import requests
import os
from dotenv import load_dotenv
from typing import Dict

load_dotenv()
TOKEN = os.getenv("LICHESS_TOKEN")

def call_api(moves: str) -> int:
    r = requests.get(f"https://explorer.lichess.org/lichess?{moves}",
                     headers={"Authorization": f"Bearer {TOKEN}"})
    
    if r.status_code != 200:
        print(f"Check authentication")  
    
    data = r.json()
    return data["white"] + data["black"] + data["draws"]

def get_priors(eco_uci: Dict[str, str]) -> Dict[str, int]:
    priors = {}
    
    # need to fill in the games
    
    
    return priors 
