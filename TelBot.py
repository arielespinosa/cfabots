#!/usr/bin/env python
import telepot
from telepot.loop import MessageLoop 
import json
import os.path
import time

class TelBot(telepot.Bot):
    def __init__(self,token):
                    
        configfile=os.path.expanduser('~/.botconfig')
        try:
            config = json.loads(open(configfile).read())

            if config['proxy']['active'] == 'yes':
                telepot.api.set_proxy(config['proxy']['url'],(config['proxy']['user'],config['proxy']['pass']))
        except:
            print('Sin Configuración de Proxy')
       
        telepot.Bot.__init__(self,token)
        # self.bot=telepot.Bot(token)
        # MessageLoop(self.bot,self.handle1)
        
    def posttry(self,msg,id):
        """ Enviar mensajes al grupo por el id """
       
        try:
            # self.sendMessage(id-1001334786762,msg,parse_mode="Markdown")
            print("Sent: "+msg)
        except:  
            print('Demasiados request. Esperando 5...')
            time.sleep(5)
            self.posttry(msg)

def handle1(msg):
    contenttype,chattype,chatid=telepot.glance(msg)
    print(contenttype,chattype,chatid)
    print(msg['text'])

if __name__ == "__main__":
    bot = TelBot("1314850663:AAFuBzMDs5niJiUXHvH6ZaWI9rXHaz7GX8A")
    print(bot.getMe())
    # print(bot.getUpdates())
    
    MessageLoop(bot,handle1).run_as_thread()
    # print("Listening")
    
    while 1:
        time.sleep(10)