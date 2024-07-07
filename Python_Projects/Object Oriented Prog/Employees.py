from abc import ABC, abstractmethod

class AbstractEmployee(ABC):
  new_id = 1
  def __init__(self):
    self.id = AbstractEmployee.new_id
    AbstractEmployee.new_id += 1
 
  @abstractmethod
  def say_id(self):
    pass
  def say_position(self):
    pass

class User:
  def __init__(self):
    self._username = None
    self._position = None

  @property
  def username(self):
    return self._username
  @property
  def position(self):
    return self._position
  
  @username.setter
  def username(self, new_name):
    self._username = new_name
    
  @position.setter
  def position(self, new_position):
    self._position = new_position

class Meeting:
  def __init__(self):
    self.attendees = []
  
  def __add__(self, employee):
    print("{} added.".format(employee.username))
    self.attendees.append(employee.username)

  def __sub__(self, employee):
    if employee.username in self.attendees:
       print("{} removed.".format(employee.username))
       self.attendees.remove(employee.username)
    else:
       print("{} is not attending the meeting.".format(employee.username))
  def __len__(self):
    return len(self.attendees)

class Employee(AbstractEmployee, User):
    def __init__(self, username, position):
      super().__init__()
      User.__init__(self)
      self.username = username
      self.position = position

    def say_id(self):
      print("My id is {}".format(self.id))
 
    def say_username(self):
      print("My username is {}".format(self.username))

    def say_position(self):
      print("My position is {}".format(self.position))