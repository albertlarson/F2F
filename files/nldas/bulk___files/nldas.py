import numpy as np
import requests
import glob
x = np.arange(0,100,1)
np.save("loaded.npy",x)
print("okay this thing is loaded")


# lis = np.load('nclistnodupes.npy')[:3]
lis = np.load('nclistnodupes.npy')



class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

    # Overrides from the library to keep headers when redirected to or from the NASA auth host.
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)
            if (original_parsed.hostname != redirect_parsed.hostname) and \
               redirect_parsed.hostname != self.AUTH_HOST and \
               original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
        return


# create session with the user credentials that will be used to authenticate access to the data

username="albertlarson"
password="Andes27Hearts34!"
session = SessionWithHeaderRedirection(username, password)



# ***********************
# Loop through Files
# ***********************
destination_files = 'ncfiles/*.nc'
y = [eeks[8:] for eeks in glob.glob(destination_files)]
not_dled = []
dled = []
for x in lis:
    if str(x[79:]) not in y:
        not_dled.append(x)
    if str(x[79:])  in y:
        dled.append(x)


# print(f'dled: \t \n \t')
# for x in dled:
#     print('\t',x,'\t')
# print(f'not dled: \t \n \t')
# for x in not_dled:
#     print('\t',x,'\t')


while True:
    for idx,i in enumerate(not_dled):
        try:

            # submit the request using the session
            response = session.get(i, stream=True)
            print(response.status_code)
            # raise an exception in case of http errors
            response.raise_for_status()
            # save the file
            with open(f'ncfiles/{i[79:]}', 'wb') as fd:
                for data in response:
                    fd.write(data)
            fd.close()
        except requests.exceptions.HTTPError as e:
            # handle any errors here
            print('error',e)
    
    dled = []        
    not_dled = []
    for x in lis:
        y = [eeks[8:] for eeks in glob.glob(destination_files)]
        if str(x[79:]) not in y:
            not_dled.append(x)
        if str(x[79:]) in y:
            dled.append(x)
    # print(f'dled: \t \n \t')
    # for x in dled:
    #     print('\t',x,'\t')
    print(f'not dled: \t \n \t')
    for x in not_dled:
        print('\t',x,'\t')

            
    if len(not_dled) == 0:
        import sys
        sys.exit('no more left to download')