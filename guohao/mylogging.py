
class MyLogger:
    
    LOGGING_LEVELS = ['ERROR', 'DEBUG', 'INFO']
    
    def __init__(self, level='DEBUG'):
        self.level = MyLogger.LOGGING_LEVELS.index(level)
        self.verbose = True
        return
    
    def set_log_level(self, level='DEBUG'):
        self.level = MyLogger.LOGGING_LEVELS.index(level)
        return
    
    def set_verbose(self, verbose=True):
        self.verbose = verbose
        return
        
    def ERROR(self, str, *args, **kwargs):
        if self.verbose and (self.level <= MyLogger.LOGGING_LEVELS.index('ERROR')):
            print(f'ERROR::{str}', *args, **kwargs)
        return

    def DEGBUG(self, str, *args, **kwargs):
        if self.verbose and (self.level <= MyLogger.LOGGING_LEVELS.index('DEBUG')):
            print(f'DEBUG::{str}', *args, **kwargs)
        return
            
    def INFO(self, str, *args, **kwargs):
        if self.verbose and (self.level <= MyLogger.LOGGING_LEVELS.index('INFO')):
            print(f'INFO::{str}', *args, **kwargs)
        return
            


# if __name__=='__main__':
#     logger = MyLogger('INFO')
#     logger.ERROR('test error')
#     logger.INFO('test info')
    