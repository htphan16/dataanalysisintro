import unicodecsv
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_csv(filename):
    with open(filename, 'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)

enrollments = read_csv('enrollments.csv')
daily_engagement = read_csv('daily_engagement.csv')
project_submissions = read_csv('project_submissions.csv')
    
### For each of these three tables, find the number of rows in the table and
### the number of unique students in the table. To find the number of unique
### students, you might want to create a set of the account keys in each table.

def numberOf(table):
    count = 0
    for i in table:
        count = count + 1
    return count
    
def numUniqueStudents(table):
    unique = set()
    for i in table:
        unique.add(i['account_key'])
    return len(unique)

for i in daily_engagement:
    i['account_key'] = i.pop('acct')


def unique_key(table):
    unique = set()
    for i in table:
        unique.add(i['account_key'])
    return unique

count = 0
for k in enrollments:
    if k['account_key'] not in unique_key(daily_engagement) and k['days_to_cancel'] != '0':
        count = count+1
        print(k)
print(count)


enrollment_num_rows = numberOf(enrollments)         # Replace this with your code
enrollment_num_unique_students = numUniqueStudents(enrollments)  # Replace this with your code

engagement_num_rows = numberOf(daily_engagement)             # Replace this with your code
engagement_num_unique_students = numUniqueStudents(daily_engagement)  # Replace this with your code

submission_num_rows = numberOf(project_submissions)             # Replace this with your code
submission_num_unique_students = numUniqueStudents(project_submissions)  # Replace this with your code

print(enrollment_num_rows)
print(enrollment_num_unique_students)
print(engagement_num_rows)
print(engagement_num_unique_students)
print(submission_num_rows)
print(submission_num_unique_students)


paid_students = dict()

for i in enrollments:
    if i['is_udacity'] == 'False':
        if i['days_to_cancel'] == '' or int(i['days_to_cancel']) > 7:
            if i['account_key'] not in paid_students.keys() \
            or dt.strptime(i['join_date'], '%Y-%m-%d') > dt.strptime(paid_students[i['account_key']],'%Y-%m-%d'):
                paid_students.update({i['account_key']:i['join_date']})


print(len(paid_students))
print(paid_students)

def within_one_week(join_date, engagement_date):
    time_delta = engagement_date - join_date
    return time_delta.days < 7 and time_delta.days >= 0

def group_data(data):
    data_group = []
    for k in daily_engagement:
        if k['account_key'] in list(paid_students.keys()):
            join_date = dt.strptime(paid_students[k['account_key']], '%Y-%m-%d')
            engagement_date = dt.strptime(k['utc_date'], '%Y-%m-%d')
            if within_one_week(join_date, engagement_date):
                data_group.append({k['account_key']:k[data]})
    return data_group


def summed_data(data):
    total_data_per_account = {}
    for k in list(paid_students.keys()):
        total_data = 0
        for i in data:
            if k in i.keys():
                total_data = total_data + float(i[k])
            total_data_per_account.update({k:total_data})
    return total_data_per_account

paid_engagements_in_first_week = group_data('utc_date')
paid_engagements_minutes = group_data('total_minutes_visited')
paid_engagements_lessons = group_data('lessons_completed')

total_minutes_per_account = summed_data(paid_engagements_minutes)
total_lessons_per_account = summed_data(paid_engagements_lessons)


total_minutes = list(total_minutes_per_account.values())
mean_minutes_per_account = np.mean(total_minutes)
max_minutes_per_account = np.max(total_minutes)
min_minutes_per_account = np.min(total_minutes)
std_minutes_per_account = np.std(total_minutes)

print("Mean:", mean_minutes_per_account)
print("Maximum:", max_minutes_per_account)
print("Minimum:", min_minutes_per_account)
print("Standard Deviation:", std_minutes_per_account)

total_lessons = list(total_lessons_per_account.values())
mean_lessons_per_account = np.mean(total_lessons)
max_lessons_per_account = np.max(total_lessons)
min_lessons_per_account = np.min(total_lessons)
std_lessons_per_account = np.std(total_lessons)

print("Mean:", mean_lessons_per_account)
print("Maximum:", max_lessons_per_account)
print("Minimum:", min_lessons_per_account)
print("Standard Deviation:", std_lessons_per_account)

def has_visited(data):
    if float(data) > 0.0:
        return 1
    else: 
        return 0


def summed_data_days(data):
    total_data_per_account = {}
    for k in list(paid_students.keys()):
        total_data = 0
        for i in data:
            if k in i.keys():
                total_data = total_data + has_visited(i[k])
            total_data_per_account.update({k:total_data})
    return total_data_per_account

paid_engagements_courses_visited = group_data('num_courses_visited')
total_days_per_account = summed_data_days(paid_engagements_courses_visited)

total_days = list(total_days_per_account.values())
mean_days_per_account = np.mean(total_days)
max_days_per_account = np.max(total_days)
min_days_per_account = np.min(total_days)
std_days_per_account = np.std(total_days)

print("Mean:", mean_days_per_account)
print("Maximum:", max_days_per_account)
print("Minimum:", min_days_per_account)
print("Standard Deviation:", std_days_per_account)

subway_project_lesson_keys = ['746169184', '3176718735']


passing_students = []
for k in project_submissions:
    if k['lesson_key'] in subway_project_lesson_keys:
        if k['assigned_rating'] == 'PASSED' or k['assigned_rating'] == 'DISTINCTION':
            passing_students.append(k['account_key'])
print(len(passing_students))

passing_engagement = []
non_passing_engagement = []
for i in paid_engagements_in_first_week:
    for k in i.keys():
        if k in passing_students:
            passing_engagement.append(i)
        else:
            non_passing_engagement.append(i)

print(len(passing_engagement))
print(len(non_passing_engagement))

minutes_per_passing_student = []
minutes_per_non_passing_student = []

for i, k in total_minutes_per_account.items():
    if i in passing_students:
        minutes_per_passing_student.append(k)
    else:
        minutes_per_non_passing_student.append(k)

lessons_per_passing_student = []
lessons_per_non_passing_student = []
for i, k in total_lessons_per_account.items():
    if i in passing_students:
        lessons_per_passing_student.append(k)
    else:
        lessons_per_non_passing_student.append(k)

days_per_passing_student = []
days_per_non_passing_student = []
for i, k in total_days_per_account.items():
    if i in passing_students:
        days_per_passing_student.append(k)
    else:
        days_per_non_passing_student.append(k)

plt.hist(minutes_per_passing_student)
plt.xlabel("total minutes visited per non passing student")
plt.show()
plt.hist(minutes_per_non_passing_student)
plt.xlabel("total minutes visited per non passing student")
plt.show()

plt.hist(lessons_per_passing_student)
plt.xlabel("total lessons completed per non passing student")
plt.show()
plt.hist(lessons_per_non_passing_student)
plt.xlabel("total lessons completed per non passing student")
plt.show()

plt.hist(days_per_passing_student, bins = 7)
plt.xlabel("total days visited per passing student")
plt.show()
plt.hist(days_per_non_passing_student, bins = 7)
plt.xlabel("total days visited per non passing student")
plt.show()

print("Mean minutes of students passed:", np.mean(minutes_per_passing_student))
print("Std minutes of students passed:", np.std(minutes_per_passing_student))
print("Max minutes of students passed:", np.max(minutes_per_passing_student))
print("Min minutes of students passed:", np.min(minutes_per_passing_student))
print("\n")
print("Mean minutes of students nonpassed:", np.mean(minutes_per_non_passing_student))
print("Std minutes of students nonpassed:", np.std(minutes_per_non_passing_student))
print("Max minutes of students nonpassed:", np.max(minutes_per_non_passing_student))
print("Min minutes of students nonpassed:", np.min(minutes_per_non_passing_student))
print("\n")


print("Mean lessons of students passed:", np.mean(lessons_per_passing_student))
print("Std lessons of students passed:", np.std(lessons_per_passing_student))
print("Max lessons of students passed:", np.max(lessons_per_passing_student))
print("Min lessons of students passed:", np.min(lessons_per_passing_student))
print("\n")
print("Mean lessons of students nonpassed:", np.mean(lessons_per_non_passing_student))
print("Std lessons of students nonpassed:", np.std(lessons_per_non_passing_student))
print("Max lessons of students nonpassed:", np.max(lessons_per_non_passing_student))
print("Min lessons of students nonpassed:", np.min(lessons_per_non_passing_student))
print("\n")



print("Mean days of students passed:", np.mean(days_per_passing_student))
print("Std days of students passed:", np.std(days_per_passing_student))
print("Max days of students passed:", np.max(days_per_passing_student))
print("Min days of students passed:", np.min(days_per_passing_student))
print("\n")
print("Mean days of students nonpassed:", np.mean(days_per_non_passing_student))
print("Std days of students nonpassed:", np.std(days_per_non_passing_student))
print("Max days of students nonpassed:", np.max(days_per_non_passing_student))
print("Min days of students nonpassed:", np.min(days_per_non_passing_student))

