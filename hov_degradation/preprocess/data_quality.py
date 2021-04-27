"""
Copyright ©2021. The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Script for generating FTP data quality plots. This script is not related to the HOV-analysis project. Written by the Connected Corridor staff
"""
import sys
import boto3
from io import BytesIO
import gzip
import matplotlib as mpl
from matplotlib import pyplot
import numpy as np
import pandas as pd
import os

end_date = input("Enter the end date (example: 2020-10-31): ")
num_days = 7

bdates = pd.date_range(end=end_date, periods=num_days).strftime('%Y%m%d')
bdates = [i for i in bdates]
access_key_id = os.environ.get('AWS_ACCESS_KEY_ID', None)
secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY', None)

# create output directory
outdir = "results_data_quality/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

def unzipper(date, time):
    # print(date+'/30sec_'+date+time+'.txt.gz')
    obj = s3res.Object('pems-raw-30sec-data',
                       date + '/30sec_' + date + time + '.txt.gz')
    n = obj.get()['Body'].read()
    gzipfile = BytesIO(n)
    gzipfile = gzip.GzipFile(fileobj=gzipfile)
    content = gzipfile.read()
    return content


def makedict(data):
    sp = data.decode().split('\n')
    name = sp[0]
    empty = {}
    for i in range(1, len(sp)):
        newsplit = sp[i].split(',')
        if newsplit[0] in combinedset:
            ind = newsplit[0]
            val = newsplit[1:]
            empty[ind] = val
    return [name, empty]


def makelist(data):
    sp = data.decode().split('\n')
    return np.array([i[0:6] for i in sp[1:]])


def sectionalizer(data, mint):
    points = mint * 2
    sectional = []
    for sensor in data:
        vdsarray = np.array([])
        for i in range(len(times) // (points)):
            try:
                if 1 in sensor[i * points:(i + 1) * points]:
                    vdsarray = np.append(vdsarray, 1)
                else:
                    vdsarray = np.append(vdsarray, 0)
            except:
                if 1 in sensor[i * points:]:
                    vdsarray = np.append(vdsarray, 1)
                else:
                    vdsarray = np.append(vdsarray, 0)
        if np.array_equal(sectional, []):
            sectional = np.array([vdsarray])
        else:
            sectional = np.append(sectional, [vdsarray], axis=0)

    sectionaltimes = len(sectional[0])
    sectionalsum = 0
    for sublist in sectional:
        sectionalsum += sum(sublist)
    sectionalmissingpercent = (1 - sectionalsum / (
    (len(combinedset) * sectionaltimes))) * 100

    def sectionalchart(data, periods, filename):
        cmap = mpl.colors.ListedColormap(['red', 'green'])
        bounds = [-1, 0.5, 2]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        pyplot.figure(figsize=(15, 5))
        pyplot.imshow(data, interpolation='nearest', cmap=cmap, norm=norm,
                      aspect='auto')
        pyplot.xticks(range(0, 2880 // periods, 120 // periods),
                      [str(x) + ':00' for x in range(0, 24)])
        pyplot.yticks(
            [len(twooneoh) - 1, +len(twooneoh) + len(onethreefour) - 1,
             len(twooneoh) + len(onethreefour) + len(sixohfive) - 1],
            ['210', '134', '605'])
        pyplot.suptitle(str(mint) + '-minute VDS availability for ' + bdate[
                                                                      0:4] + '-' + bdate[
                                                                                   4:6] + '-' + bdate[
                                                                                                6:8])
        pyplot.title('Missing Percent: ' + str(sectionalmissingpercent))
        pyplot.xlabel('Time (PST)')
        pyplot.ylabel('VDS grouped by highway')
        # img.set_size_inches(10,5)
        pyplot.savefig(filename)

    sectionalchart(sectional, points,
                   "results_data_quality/" + bdate + '_' + str(mint) + 'min_missing_chart.png')


def createchart(data, filename):
    cmap = mpl.colors.ListedColormap(['red', 'green'])
    bounds = [-1, 0.5, 2]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    pyplot.figure(figsize=(15, 5))
    pyplot.imshow(data, interpolation='nearest', cmap=cmap, norm=norm,
                  aspect='auto')
    pyplot.xticks(range(0, 2880, 120), [str(x) + ':00' for x in range(0, 24)])
    pyplot.yticks([len(twooneoh) - 1, +len(twooneoh) + len(onethreefour) - 1,
                   len(twooneoh) + len(onethreefour) + len(sixohfive) - 1],
                  ['210', '134', '605'])
    pyplot.suptitle(
        'VDS availability for ' + bdate[0:4] + '-' + bdate[4:6] + '-' + bdate[
                                                                        6:8])
    pyplot.title('Missing Percent: ' + str(missingpercent))
    pyplot.xlabel('Time (PST)')
    pyplot.ylabel('VDS grouped by highway')
    # img.set_size_inches(10,5)
    pyplot.savefig(filename)


get_last_modified = lambda obj: int(obj['LastModified'].timestamp())
s3 = boto3.client('s3')
s3res = boto3.resource('s3',
                       aws_access_key_id=access_key_id,
                       aws_secret_access_key=secret_access_key)
objs = s3.list_objects_v2(Bucket='vds-sensor-codes-bucket')['Contents']
last_added = [obj['Key'] for obj in sorted(objs, key=get_last_modified)][-1]
s3_clientobj = s3.get_object(Bucket='vds-sensor-codes-bucket', Key=last_added)
dictionaryfromVDS = eval(s3_clientobj['Body'].read().decode('utf-8'))

twooneoh = np.array(list(map(str, dictionaryfromVDS['210'])))
onethreefour = np.array(list(map(str, dictionaryfromVDS['134'])))
sixohfive = np.array(list(map(str, dictionaryfromVDS['605'])))
combined = np.concatenate((twooneoh, onethreefour, sixohfive))

twooneohset = set(twooneoh)
onethreefourset = set(onethreefour)
sixohfiveset = set(sixohfive)
combinedset = set(combined)

# tenset=set(list(map(str,dictionaryfromVDS['10'])))

# bdate = str(sys.argv[1])


for bdate in bdates:
    try:
        minutes = int(sys.argv[2])
    except:
        minutes = 5

    times = []
    for i in np.array(
            ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09'] + list(
                    map(str, range(10, 24)))):
        for j in np.array(
                ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09'] + list(
                        map(str, range(10, 60)))):
            for k in np.array(['00', '30']):
                times = np.append(times, i + j + k)

    missing = 0
    missingfiles = 0

    twooneohmissing = 0
    onethreefourmissing = 0
    sixohfivemissing = 0
    # tenmissing=0

    missinglist = []

    visual = np.array([combined])

    for time in times:
        try:
            data = unzipper(bdate, time)
        except:
            print('missingfile')
            missingfiles += 1
            missing = missing + len(combinedset)
            twooneohmissing = twooneohmissing + len(twooneohset)
            onethreefourmissing = onethreefourmissing + len(onethreefourset)
            sixohfivemissing = sixohfivemissing + len(sixohfiveset)

            visual = np.append(visual, [[False] * len(combined)], axis=0)
            continue

        listdata = makelist(data)
        smallset = set(listdata)
        visual = np.append(visual, [[x in listdata for x in combined]], axis=0)

        diff = combinedset.difference(smallset)
        twooneohdiff = twooneohset.difference(smallset)
        onethreefourdiff = onethreefourset.difference(smallset)
        sixohfivediff = sixohfiveset.difference(smallset)
        # tendiff=tenset.difference(smallset)

        missinglist = np.append(missinglist, list(diff))

        missing = missing + len(diff)
        twooneohmissing = twooneohmissing + len(twooneohdiff)
        onethreefourmissing = onethreefourmissing + len(onethreefourdiff)
        sixohfivemissing = sixohfivemissing + len(sixohfivediff)
        # tenmissing=tenmissing+len(tendiff)

    missingcounts = {}
    for i in missinglist:
        if not i in missingcounts:
            missingcounts[i] = 1
        else:
            missingcounts[i] += 1

    missingpercent = missing / (len(combinedset) * len(times)) * 100
    twooneohpercent = twooneohmissing / (len(twooneohset) * len(times)) * 100
    onethreefourpercent = onethreefourmissing / (
                len(onethreefourset) * len(times)) * 100
    sixohfivepercent = sixohfivemissing / (len(sixohfiveset) * len(times)) * 100
    # tenpercent=tenmissing/(len(tenset)*len(times))*100


    print("210-MissingPercent: " + str(twooneohpercent))
    print("134-MissingPercent: " + str(onethreefourpercent))
    print("605-MissingPercent: " + str(sixohfivepercent))
    # print("TEN-MissingPercent: "+str(tenpercent))
    print("All-MissingPercent: " + str(missingpercent))
    print("MissingFiles: " + str(missingfiles))
    print("MissingCounts: " + str(missingcounts))

    visual2 = 1 * (visual[1:] == 'True').T
    visualforcsv = np.insert(visual.T, 0,
                             np.append(['VDS'], times[0:visual.T.shape[1] - 1]),
                             axis=0)
    np.savetxt(bdate + '_missing.csv', visualforcsv, delimiter=',', header=bdate,
               comments='', fmt='%s')

    print(visual2)

    createchart(visual2, "results_data_quality/" + bdate + '_missing_chart.png')
    sectionalizer(visual2, 1)
    sectionalizer(visual2, 2)
    sectionalizer(visual2, 3)
    sectionalizer(visual2, 4)
    sectionalizer(visual2, 5)
    pyplot.close()



