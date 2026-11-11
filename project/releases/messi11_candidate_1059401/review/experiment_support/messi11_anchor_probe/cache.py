class CollectorCache:
 def __init__(self,factory):self.factory=factory;self.items={}
 def get(self,key):
  if key not in self.items:self.items[key]=self.factory(key)
  return self.items[key]
