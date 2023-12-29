#!/usr/bin/env python

from random import randint, random
import time
import copy

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# data
movies = {248576: (130, 3.3682410181274, 0), 344258: (110, 4.5819413321144, 1), 247875: (135, 24.520869061059, 0), 342183: (105, 6.3525688839221, 1), 1190376: (115, 2.3976810308718, 0), 342858: (120, 18.548330769309, 0), 346383: (110, 13.337895007968, 1), 344880: (105, 8.7481385225418, 0), 246065: (110, 8.0910852616215, 0), 1200675: (90, 3.4135008624687, 0), 1199126: (90, 4.2430678789594, 0), 344440: (90, 2.3966803710367, 0)}
movien = len(movies)
ids=[248576, 344258, 247875, 342183, 346383, 1190376, 342858, 346383, 344880, 246065, 1200675, 1199126, 344440]
hotp = np.array([movies[ids[k]][1] for k in range(movien)])
S = np.sum(hotp)
hotp /= np.sum(hotp)
halls = {'37756': (154, 1489111200, 1489158000, 1, 6), '37757': (147, 1489111200, 1489158000, 1, 6), '37758': (146, 1489111200, 1489158000, 1, 6), '37755': (235, 1489111200, 1489158000, 1, 6), '37759': (126, 1489111200, 1489158000, 1, 6), '37762': (146, 1489111200, 1489158000, 1, 6), '37754': (410, 1489111200, 1489158000, 1, 6), '37761': (186, 1489111200, 1489158000, 1, 6)}

halln = len(halls)
gtime = 1489147200
gapub = 10

def stamp2str(timeStamp):
    timeArray = time.localtime(timeStamp)
    return time.strftime("%H:%M", timeArray)

# def mymin(x):
#     m = np.min(x)
#     return m + np.mean(x-m)/(m+1)

def mymin(x):
    x = np.sort(np.unique(x))
    return np.sum(a / 30**k for k, a in enumerate(x))

class Movie(object):
    '''Movie has 4 (principal) propteries
    id_: id 
    length: length
    hot: hot
    type_: type
    '''
    
    __slots__ = ('id_', 'length', 'hot', 'type_', '__start', '__end')

    def __init__(self, id_, length, hot, type_):
        self.id_ = id_
        self.length = length * 60
        self.hot = hot / 100
        self.type_ = type_
        self.__start = 0
        self.__end = length * 60

    def __str__(self):
        if self.isgolden():
            return 'movie %d(%.4s)*: %s - %s'%(self.id_, self.hot, stamp2str(self.start), stamp2str(self.end))
        else:
            return 'movie %d(%.4s): %s - %s'%(self.id_, self.hot, stamp2str(self.start), stamp2str(self.end))

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @start.setter
    def start(self, time):
        self.__start = time
        self.__end = time + self.length

    @end.setter
    def end(self, time):
        self.__end = time
        self.__start = time - self.length

    def isgolden(self):
        return self.__end - 75 * 60 <= gtime <= self.__start + 75 * 60

    def copy(self):
        return copy.copy(self)


class Hall:
    '''Hall has 6 (principal) propteries

    id_: id 
    seat: seat
    start: start
    last: last
    entrance: entrance
    type_: type
    '''

    def __init__(self, id_, seatn, start, last, entrance=1, type_=6):
        self.id_ = id_
        self.seatn = seatn
        self.start = start
        self.last = last
        self.entrance = entrance
        self.type_ = type_
        self.movies = []
        self.admission = None
        self.manager = None
        self.min_interval = 300

    @staticmethod
    def from_db(data):
        return Hall(id_=data[0], seatn=data[2], type_=data[3])

    def __getitem__(self, key):
        for m in self.movies:
            if m.id_ == key:
                return m

    def __str__(self):
        return 'hall_%s'%self.id_

    def __repr__(self):
        return 'hall %s (%d)'%(self.id_, self.seatn)

    def random(self):
        AM = self.admission
        if AM is None:
            p = np.array([m.hot for m in self.manager.movies])
            M = stats.rv_discrete(name='movie', values=(np.arange(movien), p / np.sum(p)))
        else:
            p = np.array([self.manager.movies[k].hot for k in AM])
            M = stats.rv_discrete(name='movie', values=(AM, p / np.sum(p)))
        return M.rvs(size=1)[0]

    def dumps(self):
        print('hall %s (%d):'%(self.id_, self.seatn))
        for m in self.movies:
            if m.start <= self.last:
                print(m, end=', ')

    def insert(self, i, movie, t=0):
        self.movies.insert(i, movie)
        if i == 0:
            movie.start = h.start + t
            self.movies[i+1].start += movie.length + t
        else:
            movie.start = self.movies[i-1].start + self.min_interval + t
            self.movies[i+1].start += movie.length + self.min_interval + t
        if self.movies[i+2].start - self.movies[i+1].end < self.min_interval:
            self.movies[i+1].start = self.movies[i].end + self.min_interval
        else:
            return
        for k in range(i+1, len(self.movies)-1):
            if self.movies[k+1].start - self.movies[k].end < self.min_interval:
                self.movies[k+1].start = self.movies[k].end + self.min_interval
            else:
                return

    def append(self, movie, t=0):
        if self.movies:
            movie.start = self.movies[-1].end + t + self.min_interval
        else:
            movie.start = self.start + t
        self.movies.append(movie)

    def count(self):
        # count movies in a hall
        dict_ = {}
        for m in self.movies:
            if m.id_ in dict_:
                dict_[m.id_] += 1
            else:
                dict_.update({m.id_:1})
        return dict_

    def dist(self, other):
        k = 0
        d2 = d1 = 2100
        movies1 = [m for m in self.movies if m.start <= self.last]
        movies2 = [m for m in other.movies if m.start <= other.last]
        for m in movies1:
            for l, mm in enumerate(movies2[k:]):
                d = mm.start - m.start
                if d <= -2100:
                    k = l + 1
                    continue
                else:
                    if d <= 2100:
                        if m.id_ == mm.id_:
                            d1 = min(abs(d), d1)
                        else:
                            d2 = min(abs(d), d2)
                        k = l + 1
                    else:
                        k = l
                    break
        return d1, d2


class Criterion:
    '''Criterion has 3 (principal) propteries
    value: value
    name: name
    weight: weight [5]
    level: level [None]
    '''

    def __init__(self, value, name='', weight=5, level=None):
        self.value = value
        self.name = name
        self.weight = weight
        self.level = level

    def __repr__(self):
        return '%s: %.4f * %d'%(self.name, self.value, weight)


class Manager:
    '''Manager has 1 (principal) proptery
    halls: halls
    '''

    def __init__(self, halls, movies=None, sorted=True):
        self.halls = halls
        for h in halls:
            h.manager = self
        self.movies = movies
        if sorted:
            self.halls.sort(key=lambda h:h.seatn, reverse=True)
            if movies:
                self.movies.sort(key=lambda h:h.hot, reverse=True)
                N = sum(h.type_ for h in self.halls) + halln / 2
                for m in self.movies:
                    p = np.array([m.hot for m in self.movies])
                self.estimate = [int(k) for k in np.round(N * (p / np.sum(p)))]
        # count seat taking
        
        s = np.sum(np.sqrt(h.seatn) for h in self.halls)
        for h in self.halls:
            h.seat_rate = np.sqrt(h.seatn) / s

    @staticmethod
    def from_data(hall_data, movie_data=None):
        if movies:
            return Manager([Hall(id_, *propteries) for id_, propteries in hall_data.items()], [Movie(id_, *propteries) for id_, propteries in movie_data.items()])
        else:
            return Manager([Hall(id_, *propteries) for id_, propteries in hall_data.items()])

    @staticmethod
    def from_db(lst):
        return Manager([Hall.from_db(*args) for args in lst])

    def insert_into(self, j, k, t=None):
        if t is None:
            for time in range(gapub):
                flag = True
                for kk, h in enumerate(self.halls):
                    if kk != k:
                        for m in h.movies:
                            if time == m.start:
                                flag = False
                if flag:
                    t = time
        self.halls[k].insert(0, self.movies[j].copy(), t)

    def count(self):
        '''count movies
        dict_ : {id_: number}
        '''

        dict_ = {}
        S = 0
        for h in self.halls:
            S += len(h.movies)
            for m in h.movies:
                if m.start <= h.last:
                    if m.id_ in dict_:
                        dict_[m.id_] += 1
                    else:
                        dict_.update({m.id_:1})

        for id_ in dict_:
            dict_[id_] /= S
        return dict_

    def schedule(self, individual):
        # individual.gmovies = {}
        for k, h in enumerate(self.halls):
            n = h.type_
            h.movies = [self.movies[i].copy() for i in individual[k][1:2*n:2]]
            times = individual[k][:2*n-1:2]
            h.movies[0].start = h.start + times[0] * 300
            for l, m in enumerate(h.movies[1:], start=1):
                m.start = h.movies[l-1].end + (times[l]+1) * 300
                if m.start > h.last:
                    h.movies = h.movies[:l]
                    break

                # if m.isgolden():
                #     individual.gmovies.update({k:l})

    def initSchedule2(self, hook=list):
        # minlen = min(m.length for m in self.movies)
        individual = hook(list([] for _ in range(halln)))
        # individual.gmovies = {}
        lst = self.estimate.copy()
        i = 0
        ts = np.random.permutation(len(self.halls))
        for k, h in enumerate(self.halls):
            # golden period
            h.movies = [self.movies[i].copy()]
            h.movies[0].start = gtime - 75 * 60 + ts[k] * 300
            individual[k] = [i]
            lst[i] -= 1
            if lst[i] == 0:
                i += 1
        for k, h in enumerate(self.halls):
            # common period
            n = h.type_
            times = np.random.randint(0, gapub, size=n)
            for l in range(1, n):
                end = h.movies[0].start - (times[l]+1) * 300
                start = end - self.movies[i].length
                if h.start <= start and end <= h.movies[0].start + 300:
                    h.movies.insert(0, self.movies[i].copy())
                    h.movies[0].start = start
                    t = times[l]
                    while lst[i] <= 0:
                        i += 1
                    individual[k] = [i, t] + individual[k]
                    lst[i] -= 1
                elif start < h.start:
                    gap = (h.movies[0].start - h.start)//300
                    if gap <= gapub:
                        individual[k] = [gap] + individual[k]
                    else:
                        for j, m in enumerate(self.movies):
                            if gap * 300 - 300 * gapub <= m.length + 300 <= gap * 300 and lst[j] > 0:
                                h.movies.insert(0, m.copy())
                                h.movies[0].end = h.movies[1].start - 300
                                t0 = (h.movies[0].start - h.start)//300
                                individual[k] = [t0, i, 1] + individual[k]
                                lst[j] -= 1
                                break
                        else:
                            while lst[i] <= 0:
                                i += 1
                            lst[i] -= 1
                            t0 = randint(0, gapub-1)
                            individual[k] = [t0, i, 1] + individual[k]
                            h.movies[0].start = self.movies[i].length + 300 * (t0 +1)
                            h.movies.insert(0, self.movies[i].copy())
                            h.movies[0].start = t0 * 300
                            for l in range(1, len(h.movies)-1):
                                m, mm = self.movies[l], self.movies[l+1]
                                if m.end <= mm.start - 300:
                                    break
                                else:
                                    mm.start = m.end + 300
                                individual[k][l*2+2] = 1
                    break

                    # if h.movies[-1].isgolden():
                    #     individual.gmovies.update({k:l})
                    # break

            t = times[-1]
            start = h.movies[-1].end + t * 300
            if start <= h.last:
                while lst[i] <= 0:
                    i += 1
                h.movies.insert(0, self.movies[i].copy())
                h.movies[-1].start = start
                individual[k] = individual[k] + [t, i]
                lst[i] -= 1
            d = h.type_ - len(h.movies)
            if d > 0:
                for _ in range(d):
                    if h.movies[-1].end + 300 <= h.last:
                        h.append(self.movies[i].copy())
                    individual[k] = individual[k] + [1, i]
            elif d < 0:
                individual[k] = individual[k][:2*d]
                h.movies = h.movies[:d]
        return individual

    def initSchedule1(self, hook=list):
        # minlen = min(m.length for m in self.movies)
        individual = hook(list([] for _ in range(halln)))
        # individual.gmovies = {}
        lst = self.estimate.copy()
        i = 0
        ts = np.random.permutation(len(self.halls))
        for k, h in enumerate(self.halls):
            # Arrange movies in prime time
            h.movies = [self.movies[i].copy()]
            h.movies[0].start = gtime - 75 * 60 + ts[k] * 300
            individual[k] = [i]
            lst[i] -= 1
            if lst[i] == 0:
                i += 1
            # Arrange movies in common time
            n = h.type_
            times = np.random.randint(0, gapub, size=n)
            for l in range(1, n):
                end = h.movies[0].start - (times[l]+1) * 300
                start = end - self.movies[i].length
                if h.start <= start and end <= h.movies[0].start + 300:
                    h.movies.insert(0, self.movies[i].copy())
                    h.movies[0].start = start
                    t = times[l]
                    while lst[i] <= 0:
                        i += 1
                    individual[k] = [i, t] + individual[k]
                    lst[i] -= 1
                elif start < h.start:
                    gap = (h.movies[0].start - h.start)//300
                    if gap <= gapub:
                        individual[k] = [gap] + individual[k]
                    else:
                        for j, m in enumerate(self.movies):
                            if gap * 300 - 300 * gapub <= m.length + 300 <= gap * 300 and lst[j] > 0:
                                h.movies.insert(0, m.copy())
                                h.movies[0].end = h.movies[1].start - 300
                                t0 = (h.movies[0].start - h.start)//300
                                individual[k] = [t0, i, 1] + individual[k]
                                lst[j] -= 1
                                break
                        else:
                            while lst[i] <= 0:
                                i += 1
                            lst[i] -= 1
                            t0 = randint(0, gapub-1)
                            individual[k] = [t0, i, 1] + individual[k]
                            h.movies[0].start = self.movies[i].length + 300 * (t0 +1)
                            h.movies.insert(0, self.movies[i].copy())
                            h.movies[0].start = t0 * 300
                            for l in range(1, len(h.movies)-1):
                                m, mm = self.movies[l], self.movies[l+1]
                                if m.end <= mm.start - 300:
                                    break
                                else:
                                    mm.start = m.end + 300
                                individual[k][l*2+2] = 1
                    break

                    # if h.movies[-1].isgolden():
                    #     individual.gmovies.update({k:l})
                    # break

            t = times[-1]
            start = h.movies[-1].end + t * 300
            if start <= h.last:
                while lst[i] <= 0:
                    i += 1
                h.movies.insert(0, self.movies[i].copy())
                h.movies[-1].start = start
                individual[k] = individual[k] + [t, i]
                lst[i] -= 1
            d = h.type_ - len(h.movies)
            if d > 0:
                for _ in range(d):
                    if h.movies[-1].end + 300 <= h.last:
                        h.append(self.movies[i].copy())
                    individual[k] = individual[k] + [1, i]
            elif d < 0:
                individual[k] = individual[k][:2*d]
                h.movies = h.movies[:d]
        return individual

    def initSchedule(self, hook=list):
        if random() < .5:
            return self.initSchedule1(hook)
        else:
            return self.initSchedule2(hook)

    def fitness(self):
        return self.time_interval(), self.check_rate(), self.total_hot(), self.check_time()

    def check(self):
        d1, d2 = self.check_interval()
        print('''
Minimum time interval: %.4f+%.4f;
Similarity between popularity and show times: %.4f;
Total popularity (prime time): %.4f(%.4f);
The number of full-screen movie halls: %d'''%(d1, d2, self.check_rate(), self.total_hot(), self.ghot(), self.check_time()))

    def print_fitness(self):
        print('fitness: %.4f, %.4f, %.4f, %d'%(self.time_interval(), self.check_rate(), self.total_hot(), self.check_time()))

    def hot(self):
        # total popularity
        return sum(sum(m.hot for m in h.movies if m.start<=h.last) * h.seatn for h in self.halls)

    def ghot(self):
        # prime time
        hot = 0
        for h in self.halls:
            for m in h.movies:
                if m.isgolden():
                    hot += m.hot * h.seatn
                    break
        return hot

    def total_hot(self):
        # Weighted popularity
        return sum(sum(m.hot for m in h.movies if m.start<=h.last) * h.seatn for h in self.halls) + 3 * self.ghot()

    def check_time(self):
        # check time-out
        N = 0
        for h in self.halls:
            if h.movies[-1].start <= h.last:
                N +=1
        return N

    def check_rate(self):
        """Popularity ~ Times ratio ~ Screening rate ~ Number ratio ~ Box office rate
The degree of similarity between the system recommended screening rate and the actual screening rate
        """
        dict_ = self.count()
        d = 0
        for id_, rate in dict_.items():
            d += abs(movies[id_][1]/S - rate)
        return 1 / (d + 0.001)

    def check_interval(self):
        # opening interval
        d1s = []
        d2s = []
        for k, h in enumerate(self.halls[:-1]):
            for hh in self.halls[k+1:]:
                d1, d2 = h.dist(hh)
                d1s.append(d1)
                d2s.append(d2)
        return min(d1s) / 60, min(d2s) / 60

    def time_interval(self):
        # opening interval
        deltas = []
        for k, h in enumerate(self.halls[:-1]):
            for hh in self.halls[k+1:]:
                d1, d2 = h.dist(hh)
                deltas.append((d1*0.5 + d2*0.5))
        delta = mymin(deltas)

        return delta / 60

    def criterion1(self):
        # Rationality of arranging movie screening halls(安排影片放映厅的合理性)
        c = self.count()
        alpha = 0
        for m in self.movies:
            for k, h in enumerate(self.halls):
                hc = h.count()
                if m.id_ in hc and c[m.id_] < hc[m.id_] * 2:
                    alpha += abs(m.hot - h.seat_rate)
                    break
        return alpha

    def criterion2(self):
        # The degree of similarity between the system recommended screening rate and the actual screening rate(系统推荐排片率与实际排片率接近程度)
        return self.check_rate()

    def criterion3(self):
        # the number of movies shown during the prime time(黄金时间段放映电影数)
        hot = 0
        for h in self.halls:
            for m in h.movies[::-1]:
                if m.isgolden():
                    hot += 1
                    break
        return hot

    def criterion4(self):
        # the most popular movie screened in the optimal hall during the prime time(最火的影片排入最优厅黄金时间段)
        for m in self.halls[0].movies:
            if m.id_ == self.movies[0].id_ and m.isgolden():
                return 1
        return 0

    # def golden(self):
    #     c = {}
    #     for h in self.halls:
    #         for m in h:
    #             if m.isgolden():
    #                 if m.id_ in c:
    #                     c[m.id_].append(m)
    #                 else:
    #                     c[m.id_] = [m]
    #     return c

    def criterion5(self):
        return 1

    def criterion6(self):
        # Rationality of the interval between the opening of all movies in prime time (所有电影黄金时段开映间隔合理性)
        times = np.sort([m.start for h in self.halls for m in h.movies if m.isgolden()])
        times = np.diff(times)
        return 1

    def criterion7(self):
        # (所有电影非黄金时段开映间隔合理性)
        times = np.sort([m.start for h in self.halls for m in h.movies if not m.isgolden()])
        times = np.diff(times)
        return 1

    def criterion8(self):
        # (避免同时开场)
        return 1

    def criterion9(self):
        # (高票房日子场间隔尽量短)
        times = np.sort([m.start for h in self.halls for m in h.movies])
        return 1

    def criterion10(self, latest='22'):
        # (低热度动画片开映时间合理性)
        n = 0
        for h in self.halls:
            for m in hall:
                if '动画' in m.type and m.hot < 1/halln and m.end > latest:
                    n += 1
        return n

    def criterion11(self, earliest='22'):
        # (低热度动画片开映时间合理性)
        n = 0
        for h in self.halls:
            for m in hall:
                if '恐怖' in m.type and m.hot < 0.5/halln and m.start < earliest:
                    n += 1
        return n

    def hasbighall(self):
        return self.halls[0].seatn > self.halls[1].seatn * 1.5

    def criterion12(self):
        # Hall sharing status(大厅共用情况)
        m, mm = self.movies[:2]
        if self.hasbighall() and abs(m.hot - mm.hot) < 0.05:
            gm = [m for m in self.halls[0].movies() if m.isgolden]
            if set(m.id_ for m in gm) == {m.id_, mm.id_}:
                return 1
            else:
                return 0


    def criterion13(self):
        # The richness of screening(影片排映丰富度)
        if halln >= 6 and sum(m.hot for m in self.movies[:5])> .05:
            return len(self.count())


    def criterion14(self, minhot=0.1):
        # not popular movie (小片不在黄金时段)
        n = 0
        for h in self.halls:
            for m in h:
                if m.isgolden() and m.hot < minhot:
                    n += 1
        return n

    def print_criterion(self):
        for k in range(1, 13):
            if k != 10 and k!=11:
                print('criterion%d:'%k, getattr(self, 'criterion%d'%k)())

    def dumps(self):
        for h in self.halls:
            h.dumps()
            print()

    def stat(self):
        dict_ = self.count()
        for id_, rate in dict_.items():
            print(movies[id_][1]/100, rate)

    def plot(self, axes=None):
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        axes.invert_yaxis()

        def format_fn(tick_val, tick_pos):
            return stamp2str(tick_val)
        axes.xaxis.set_major_formatter(FuncFormatter(format_fn))
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        def format_fn(tick_val, tick_pos):
            k = int(tick_val)
            if k < len(self.halls):
                h = self.halls[k]
                return '%s(%d)'%(h.id_, h.seatn)
            else:
                return ''
        axes.yaxis.set_major_formatter(FuncFormatter(format_fn))
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))

        H = len(self.halls)
        for k, h in enumerate(self.halls):
            for m in h.movies:
                if m.hot > 0.2:
                    color = 'r'
                elif m.hot > 0.15:
                    color = 'y'
                elif m.hot > 0.1:
                    color = 'g'
                elif m.hot > 0.05:
                    color = 'c'
                else:
                    color = 'b'
                axes.text(m.start, k, '%d'%m.id_)
                axes.plot((m.start, m.end), (k, k), color=color, linestyle='-')
            axes.plot((h.start, h.start), (k-1/2, k+1/2), color='k')
            axes.plot((h.last, h.last), (k-1/2, k+1/2), color='k')
        axes.plot((gtime-75*60, gtime-75*60), (0, H), color='y', linestyle='--')
        axes.plot((gtime+75*60, gtime+75*60), (0, H), color='y', linestyle='--')
        axes.set_xlabel('time')
        axes.set_ylabel('hall')
        axes.set_title('movie schedule')
        plt.show()


manager = Manager.from_data(halls, movies)


from deap import tools

def mutRandom(individual, indpb1, indpb2):
    ts = np.random.permutation(gapub)
    for k, hall in enumerate(individual):
        hall[0] = ts[k]
    for k, hall in enumerate(individual):
        if random() < indpb1:
            for i in range(1, len(hall), 2):
                if random() < indpb2:
                    if random() < 0.7:
                        if hall[i] == 0:
                            hall[i] += 1
                        else:
                            hall[i] -= 1
                    else:
                        if hall[i] == movien:
                            hall[i] -= 1
                        else:
                            hall[i] += 1

        else:   
            for i in range(2, len(hall)-1, 2):
                if random() < indpb2:
                    hall[i] = np.random.choice([t for t in range(gapub) if t != hall[i]])
        h = randint(0, halln-2)
        if random() < 0.3:
            individual[h], individual[h+1] = individual[h+1], individual[h]
        else:
            individual[h], individual[h+1] = tools.cxTwoPoint(individual[h+1], individual[h])
        return individual


from pyrimidine import BaseIndividual, HOFPopulation
from pyrimidine.deco import side_effect, fitness_cache


class Chromosome(list):

    def copy(self, type_=None):
        return copy.deepcopy(self)

    def cross(self, other):
        k = randint(1, len(self)-1)
        return self.__class__(np.concatenate((self[:k], other[k:]), axis=0))


@fitness_cache
class Individual(BaseIndividual):

    element_class = Chromosome

    @side_effect
    def mutate(self):
        self[:] = mutRandom(self, indpb1=0.15, indpb2=0.8)

    def cross(self, other):
        s1 = set(h[0] for h in self)
        s2 = set(h[0] for h in other)
        if random() > 1/(len(s1.symmetric_difference(s2))+1):
            return super().cross(other)
        else:
            return self.copy()

    def _fitness(self):
        manager.schedule(self)
        return np.dot((50, 20, 2, 1), manager.fitness())


Population = HOFPopulation[Individual]

if __name__ == '__main__':

    pop = Population([manager.initSchedule() for _ in range(50)])
    pop.evolve()
    ind = pop.best_individual

    manager.schedule(ind)
    manager.print_fitness()

    manager.check()
    manager.dumps()
    manager.plot()
    manager.print_criterion()
