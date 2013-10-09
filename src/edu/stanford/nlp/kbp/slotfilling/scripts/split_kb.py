#!/usr/bin/env python

import os
import sys

# returns a list of entity text from the .xml file
def ReadFile(filename):
    entities = []
    lines = open(filename).readlines()
    current_entity = None
    for line in lines:
        if line.startswith("<entity "):
            current_entity = []
            current_entity.append(line)
        elif line.startswith("</entity>"):
            current_entity.append(line)
            entities.append(current_entity)
            current_entity = []
        elif current_entity != None:
            current_entity.append(line)

    return entities

# returns a list of integer bounds
# 0 at the start, length at the end, enough fenceposts between
# to make divisions roughly equal intervals
def CalculateBounds(length, divisions):
    bounds = []
    for i in xrange(divisions):
        bounds.append(length * i / divisions)
    bounds.append(length)
    return bounds

# returns a list of lists
# the returned lists are the original list, split into divisions
# pieces as evenly as possible
def SplitList(original_list, divisions):
    bounds = CalculateBounds(len(original_list), divisions)
    results = []
    for i in xrange(divisions):
        results.append(original_list[bounds[i]:bounds[i+1]])
    return results


# given a list of "entities", in other words a list of lines (ending
# in newline) which comprise an entity, prints them out to the given
# filename.  wraps this in the <knowledge_base> tag needed by kbp
def OutputEntitiesFile(entities, filename):
    fout = open(filename, "w")
    fout.write("<?xml version='1.0' encoding='UTF-8'?>\n<knowledge_base>\n")
    for entity in entities:
        for line in entity:
            fout.write(line)
    fout.write("</knowledge_base>\n")
    fout.close()

# reads the entities in filename, splits it evenly into divisions
# files, and writes it back into files with "part#" added to the
# parts.  parts count from 0 to divisions-1.  if path != None, then
# parts are output to that directory instead of the original directory.
def SplitEntitiesFile(filename, divisions, path = None):
    entities = ReadFile(filename)
    split_entities = SplitList(entities, divisions)
    if filename.endswith(".xml"):
        basefilename = filename[:-4]
    else:
        basefilename = filename
    if path:
        basefilename = os.path.join(path, os.path.basename(basefilename))
    for i in xrange(divisions):
        newfilename = "%s.part_%d_of_%d.xml" % (basefilename, i, divisions)
        OutputEntitiesFile(split_entities[i], newfilename)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        SplitEntitiesFile(sys.argv[1], int(sys.argv[2]))
    elif len(sys.argv) == 4:
        SplitEntitiesFile(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    else:
        print "Expected arguments: " 
        print "  " + sys.argv[0] + " <file> <splits> [path]"

