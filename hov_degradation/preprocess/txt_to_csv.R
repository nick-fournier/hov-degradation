# """
# Copyright ©2021. The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.
# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# """

library(data.table)

indir = "C:\\git_clones\\connected_corridors\\hov-degradation\\experiments\\D7\\data\\"
outdir = "C:\\git_clones\\connected_corridors\\hov-degradation\\experiments\\district_7\\data\\"

#
headers = colnames(fread(paste0(outdir,"station_5min_2020-05-24.csv"), nrows = 1))
headers[headers=="V1"] <- ""


#
flist = list.files(indir)
flist = flist[grepl("station_5min", flist)]

#
for(file in flist) {
  tmp = fread(paste0(indir, file))
  tmp = cbind(1:nrow(tmp), tmp)
  colnames(tmp) <- headers
  
  #
  fout = gsub("d07_text_", "", file)
  fout = gsub("_","-", fout)
  fout = gsub("-5min-", "_5min_", fout)
  fout = gsub(".txt", ".csv", fout)
  #
  fwrite(tmp, file = paste0(outdir, fout))
  print(paste("Done with", file))
}
