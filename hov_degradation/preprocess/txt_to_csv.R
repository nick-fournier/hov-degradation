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
