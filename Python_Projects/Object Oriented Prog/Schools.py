class School:
  def __init__(self, name, level, numberOfStudents):
    if level not in ["primary", "middle", "high"]:
            raise TypeError("Invalid level. Please choose from 'primary', 'middle', or 'high'.")
    else:
      self.name = name
      self.level = level
      self.numberOfStudents = numberOfStudents
  
  def getName(self):
    return self.name
  def getlevel(self):
    return self.level
  def getnumberOfStudents(self):
    return self.numberOfStudents

  def setnumberOfStudents(self, number):
    self.number = self.numberOfStudents

  def __repr__(self):
    return ("A {} school name {} with {} students".format(self.level, self.name, self.numberOfStudents))
  
s1 = School("Teresa", "middle", 24)

print(s1.getName())
print(s1.getlevel())
print(s1.getnumberOfStudents())
print(s1)
s1.numberOfStudents = 350
print(s1)

class PrimarySchool(School):
  def __init__(self, name, numberOfStudents, pickupPolicy):
    super().__init__(name, "primary", numberOfStudents)
    self.pickupPolicy = pickupPolicy
  def getpickupPolicy(self):
    return self.pickupPolicy
  def __repr__(self):
    super().__repr__()
    return ("The School policy is : {}".format(self.pickupPolicy))

ps1 = PrimarySchool("David", 500, "no uniforms")
print(ps1)
print(ps1.getpickupPolicy())
print(ps1.getnumberOfStudents())

class HighSchool(School):
  def __init__(self, name, numberOfStudents, sportsTeams):
    super().__init__(name, "high", numberOfStudents)
    self.sportsTeams = sportsTeams
  def getsportsTeams(self):
    return self.sportsTeams
  def __repr__(self):
    super().__repr__()
    return ("The sports teams are: {}").format(self.sportsTeams)

hs1 = HighSchool("Oak", 790, ["football", "basketball", "tennis"])

print(hs1.getsportsTeams())
print(hs1)