import requests
from tqdm import tqdm
import time

def web_search(smiles_list, save=False, libary="PubChem"):
    """Search for molecules in online databases. 

    Args:
        smiles_list (list[str]): List of SMILES strings to be searched for in an online database. 
        save (bool, optional): Save the molecules found in online databases in a file together with the url leading to the database entry. Defaults to False.
        libary (str, optional): Specifies the online database. Defaults to "PubChem".
    """

    return

def search_pubchemold(smiles_list):
    #SMILES = "C1=CC=CC=C1ccccccccccccccccccccccccccccccccccccccccccccccccN"
    # SMILES = "C1=CC=CC=C1"
    
    success_count = 0
    matches = {

        }
    failures = []
    found = []
    for i, smiles in enumerate(tqdm(smiles_list)):
        #url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
        data=smiles
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.get(url, headers=headers, data=data) 
        success = False
        try:
            json_data = response.json()
            cid = json_data.get("IdentifierList", {}).get("CID", [0])[0]
        except:
            cid = 0
        if cid != 0:
            matches[smiles] = [f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/JSON", cid]
            #print(f"{i} Success: {smiles}") 
            success_count+=1
            success = True
        else:
            failures.append(smiles)
        found.append(success)
    print()
    print(f"# of molecules found on PubChem: {success_count}")
    print(f"# of molecules not found on PubChem: {len(failures)}")
    print(f"Successrate: {(success_count/len(smiles_list))*100}%")
    return matches, failures, found


def parse_throttling_header(header_value):
    if header_value == "":
        return {
            'Request Count':["Unknown", 0],
            'Request Time':["Unknown", 0],
            'Service':["Unknown", 0]
        }
    statuses = dict()
    parts = header_value.split(",")
    for part in parts:
        key,value = part.strip().split(" status: ") 
        status, percent= value.split(" (")
        percent = percent[:-2]
        statuses[key] = [status, percent]
    return statuses


def get_delay(thrott_status):
    worst = "Green"
    for key, value in thrott_status.items():
        if value[0] == "Red":
            return 60 # 1 minute
        elif value[0] == "Yellow":
            worst = "Yellow"
    if worst == "Yellow":
        return 1 # 1 second
    return 0.2
        


def search_pubchem(smiles_list):
    #SMILES = "C1=CC=CC=C1ccccccccccccccccccccccccccccccccccccccccccccccccN"
    # SMILES = "C1=CC=CC=C1"
    # {"Request Count": "?", "Request Time": "?", "Service": "?"}
    success_count = 0
    matches = {

        }
    failures = []
    found = []
    progress = tqdm(smiles_list)
    
    for i, smiles in enumerate(progress):
        success = False

        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
        retry = 0
        while retry>=0:
            try:
                response = requests.get(url) 
                break
            except requests.exceptions.RequestException as e:
                retry +=1 
                wait_time = 60 * (retry)
                print(f"Sleep for {round(wait_time/60,2)} min")
                time.sleep(wait_time)
                


        thrott_status = parse_throttling_header(response.headers.get('X-Throttling-Control', ""))
        progress.set_postfix({
            "Count": f"{thrott_status['Request Count'][1]}% ({thrott_status['Request Count'][0]})",
            "Time": f"{thrott_status['Request Time'][1]}% ({thrott_status['Request Time'][0]})", 
            "Service": f"{thrott_status['Service'][1]}% ({thrott_status['Service'][0]})",
            "Status": response.status_code
            })#" ".join([f"{key}: {value[1]}% ({value[0]})" for key, value in thrott_status.items()]))
        
        delay = get_delay(thrott_status)
        if response.status_code == 200 or response.status_code == 202:
            try:
                json_data = response.json()
                cid = json_data.get("IdentifierList", {}).get("CID", [0])[0]
            except:
                cid = 0

        else:
            cid=0

        if cid != 0:
            matches[smiles] = [f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/JSON", cid]
            #print(f"{i} Success: {smiles}") 
            success_count+=1
            success = True
        else:
            failures.append(smiles)
        found.append(success)
        time.sleep(delay)
    print()
    print(f"# of molecules found on PubChem: {success_count}")
    print(f"# of molecules not found on PubChem: {len(failures)}")
    print(f"Successrate: {(success_count/len(smiles_list))*100}%")
    return matches, failures, found



