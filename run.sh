#!/bin/sh
HEAP=20g

time java -ea -cp classes/:lib/commons-compress-1.3.jar:lib/commons-lang-2.5.jar:lib/fastutil.jar:lib/faust-gazetteer-1.0.3003.jar:lib/htmlparser.jar:lib/jgraph.jar:lib/jgrapht.jar:lib/joda-time.jar:lib/lucene-core-3.0.3.jar:lib/protobuf-java-2.3.0.jar:lib/ra.jar:lib/stanford-corenlp-2012-05-22-models.jar:lib/stanford-corenlp-2012-05-22.jar:lib/xom.jar -Xmx$HEAP -XX:MaxPermSize=512m $@

