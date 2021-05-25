ARCHIVE_NAME=./ent15_classifTable.txt

# Upload the built library onto transfer.sf
PACKAGE_LOCATION=$(curl -F"file=@${ARCHIVE_NAME}" https://file.io)

if [[ $PACKAGE_LOCATION =~ \"success\":true,.*\"link\":\"(.+)\",.* ]]
then
	LINK="${BASH_REMATCH[1]}"
else
	exit 255
fi

echo $LINK
