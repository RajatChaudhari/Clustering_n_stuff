import pyhdb
import json 
import yaml

cfg = yaml.load(open("config.yml", 'r'))['hana']

def dump_to_json_store(json_data):
    try:
        json_dump = json.dumps(json_data)
        json_str = json_dump.replace("\'s" ,"").replace("s\'" ,"")
        connection = pyhdb.connect(host=cfg['host'],port=cfg['port'],user=cfg['user'],password=cfg['password'])
        cursor = connection.cursor()
        cursor.execute('SET SCHEMA "SCHEMAJPA";')
        cursor.execute("INSERT INTO JPA_PP_JP_CLEAN_COLLECTION VALUES('" + json_str + "');")
        connection.commit()
        return True
    except:
        return False
        
    