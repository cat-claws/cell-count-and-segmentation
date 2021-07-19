rm -rf CoNSeP
# wget https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep.zip > /dev/null
python util_runtime.py
unzip ./consep.zip > /dev/null
rm -rf _*; rm consep.zip
