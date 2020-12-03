import os
import sys

def Suffix(tree):
	tree = [i.split(".")[1] for i in tree]
	return max(tree)

def update():
	work_path = sys.path[0] +"/"
	folders = []
	tree = {}

	dirs = os.scandir(work_path)
	for i in dirs:
		if i.is_dir():
			name = str(i.name)
			if "-" in name:
				if name.split("-")[1] == "model":
					folders.append(name)
		

	with open("models.list", "w") as lists:
		with open("models.py", "w") as models:
			models.write("work = \"%s\"\n"% work_path)
			models.write("suffix = {\n")
			for i in folders:
				tree[i] = os.listdir(work_path+i)
				models.write("\t\"%s\": \"%s\", \n"%(Suffix(tree[i]), i))
			models.write("}\ndlmodels = {\n")
			for i in folders:
				lists.write("%s\n"%i)
				models.write("\t\"%s\": [\"%s\"],\n"%(i, "\", \"".join(tree[i])))
				for x in tree[i]:
					lists.write("\t%s\n"%x)
			models.write("}")

if __name__ == "__main__":
	update()
