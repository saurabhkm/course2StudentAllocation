import pandas as pd
import numpy as np
import utils
import argparse

parser = argparse.ArgumentParser(description='MCMC based teaching assistant allocation')
parser.add_argument('-coursesRequirements', type=str, default='../data/courseFile3.csv', help='Path to courses requirements file')
parser.add_argument('-studentPreferences', type=str, default='../data/studentsWithCpi.csv', help='Path to student preferences file')
parser.add_argument('-costWeights', nargs='+', type=float, default=[-1.0, 1.0, 1.0], help='Weightages for preference, previous grade and CPI distribution')
parser.add_argument('-choiceIndices', nargs='+', type=float, default=[0, 1, 2, 3, 4, 5],help='Weightages for preference, previous grade and CPI distribution')
parser.add_argument('-choiceWeights', nargs='+', type=float, default=[10, 1, 2, 3, 4, 5], help='Weightages for preference, previous grade and CPI distribution')
parser.add_argument('-seed', type=float, default=2**13 - 1, help='Path to student preferences file')
parser.add_argument('-iterations', type=int, default=45000, help='Path to student preferences file')
args = parser.parse_args()

np.random.seed(args.seed)
nIters = args.iterations

class allotment(object):
    """An allotment class"""
    def __init__(self, studentCount, courseCount, times):
        super(allotment, self).__init__()
        self.studentCount = studentCount
        self.courseCount = courseCount
        self.times = times
        self.data = np.eye(studentCount)
        self.squeeze()
    
    def squeeze(self):
        dataHolder = np.zeros((self.studentCount, self.courseCount))
        start = 0
        for i in range(self.courseCount):
            stop = start + self.times[i]
            dataHolder[:,i] = (np.sum(self.data[:,start:stop], 1)).ravel()
            start = stop
        self.data = np.array(dataHolder, dtype=np.int64)

    def calcGoodness(self, choiceWeights, cpiArray):
        a = self.data*choiceWeights
        b = np.ma.masked_where(a == 0, a)
        choiceGoodness = np.array(np.nanmean(b, 1), dtype=np.float)
        cpiRepeated = (np.array([cpiArray, ]*self.courseCount)).transpose()
        SvC_cpi = self.data * cpiRepeated
        b = np.ma.masked_where(SvC_cpi == 0, SvC_cpi)
        cpiGoodness = np.array(np.nanmean(b, 0), dtype=np.float)
        return choiceGoodness, cpiGoodness

    def swapRows(self):
        swapId1 = np.random.randint(0, self.studentCount-1)
        swapId2 = np.random.randint(0, self.studentCount-1)
        temp = self.data.copy()
        tempVar = temp[swapId1,:].copy()
        temp[swapId1,:] = temp[swapId2,:]
        temp[swapId2,:] = tempVar
        return temp

coursesDF = pd.read_csv(args.coursesRequirements)
studentsDF = pd.read_csv(args.studentPreferences)

courseCount = len(coursesDF.index)
studentCount = len(studentsDF.index)
cpiArray = studentsDF['CPI']
times = np.array(coursesDF['CourseNeeds'])

choiceIdx = utils.makeArray(studentCount, studentsDF, courseCount, coursesDF, args.choiceIndices)
choiceWeights = utils.makeArray(studentCount, studentsDF, courseCount, coursesDF, args.choiceWeights)

initialAllotment = allotment(studentCount, courseCount, times)

choiceGoodnessOld, cpiGoodnessOld = initialAllotment.calcGoodness(choiceWeights, cpiArray)

finalAllottment, utility = utils.runMCMC(initialAllotment, nIters, studentCount, courseCount, args.costWeights, choiceIdx, choiceWeights, studentsDF, cpiArray)

choiceGoodnessNew, cpiGoodnessNew = finalAllottment.calcGoodness(choiceWeights, cpiArray)

utils.writePerformance(finalAllottment, choiceGoodnessOld, cpiGoodnessOld, choiceGoodnessNew, cpiGoodnessNew, utility)