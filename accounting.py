from __future__ import division

import time


class Task(object):

    def __init__(self, name, start_time=time.time(), end_time=None):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time

    def finish(self):
        self.end_time = time.time()

    @property
    def is_finished(self):
        return self.end_time != None

    @property
    def total_time(self):
        if not self.is_finished:
            return float('inf')
        return self.end_time - self.start_time

    def assert_same_type(self, other):
        assert self.name == other.name, "Can't compare %s and %s" % (
            self.name, other.name
        )

    def __lt__(self, other):
        self.assert_same_type(other)
        this = self.total_time
        that = other.total_time
        return this < that

    def __eq__(self, other):
        return self.name == other.name and \
            self.start_time == other.start_time and \
            self.end_time == other.end_time

    def __add__(self, other):
        print "add", self, other
        self.assert_same_type(other)
        return Task(self.name, 0, self.total_time + other.total_time)

    def __str__(self):
        return "Task {0} ({1}) {2}".format(
            self.name, self.is_finished, self.total_time
        )


class Accounting(object):

    bookkeeping = {}

    @classmethod
    def start_task(cls, task_name):
        task_list = cls.bookkeeping.get(task_name, [])
        task_list.append(Task(task_name))
        cls.bookkeeping[task_name] = task_list

    @classmethod
    def finish_task(cls, task_name):
        assert task_name in cls.bookkeeping, 'No task `%s`' % task_name
        task_list = cls.bookkeeping[task_name]
        last_task = task_list[-1]
        assert not last_task.is_finished, 'No running task `%s`' % task_name
        last_task.finish()

    @classmethod
    def print_stats(cls):
        print "\nExecution statistics:\n"
        bk = cls.bookkeeping
        for task in sorted(bk.keys()):
            task_list = bk[task]
            sum_task_list = sum(t.total_time for t in task_list)
            print "Task %s" % task
            print "  Times executed:", len(task_list)
            print "  Min time: {0:.4f}s".format(min(task_list).total_time)
            print "  Max time: {0:.4f}s".format(max(task_list).total_time)
            print "  Avg time: {0:.4f}s".format(sum_task_list / len(task_list))
            print "  Total time spent: {0:.4f}s".format(sum_task_list)
            print ""
