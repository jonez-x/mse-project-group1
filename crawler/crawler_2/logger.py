import datetime
import asyncio
import os

class Logger:
    def __init__(self):
        self.log_file = "crawl_info/log-file.txt"
        self.sema_log = asyncio.Semaphore()

    def create_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("Created File. \n")
                pass

    async def print_msg(self, msg, type):
        async with self.sema_log:
            with open(self.log_file, "a") as f:
                match type:
                    case "i":
                        f.write("[INFO]     " + str(datetime.datetime.now()) + " " + msg + "\n")
                    case "w":
                        f.write("[WARNING]  " + str(datetime.datetime.now()) + " " + msg+ "\n")
                    case "e":
                        f.write("[ERROR]    " + str(datetime.datetime.now()) + " " + msg+ "\n")