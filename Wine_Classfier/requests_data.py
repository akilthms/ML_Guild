from ftplib import FTP
import requests
#import data
url = 'https://deloitteus.sharefile.com/d-sfe252a894de49d08'
def get_data(url,user,pwd,reqType='requests'):
    if reqType == 'requests':
        r = requests.get(url, auth=(user, pwd))
        print("The response for the given request is {}".format(r.status_code))
        #print(r.json()) 
        return r
    elif reqType.lower() == 'ftp':
        ftp = FTP(url)
        ftp.login(user,pwd)
        ftp.retrlines('LIST')
        ftp.quit()

r=get_data(url,'akthomas','Datascience823!')
rftp = get_data(url,'akthomas','Datascience823!','ftp')