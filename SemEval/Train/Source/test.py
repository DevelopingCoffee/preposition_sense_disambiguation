class Solution:
    def checkIfPrerequisite(self, n, prerequisites, queries):
        kanten = {}
        for e in prerequisites:
            if e[0] in kanten:
                kanten[e[0]].add(e[1])
            else:
                kanten[e[0]] = set()
                kanten[e[0]].add(e[1])


        ret = []

        for e in queries:
            besucht = set()
            if e[0] not in kanten:
                ret.append(False)
                continue

            queue = list(kanten[e[0]])
            while True:
                print(queue)
                queue2 = set()
                for q in queue:
                    if q not in besucht:
                        if q in kanten:
                            queue2 = queue2.union(kanten[q])
                            print(kanten)
                        besucht.add(q)
                queue = list(queue2)
                print(queue)
                if e[1] in besucht:
                    ret.append(True)
                    break
                if queue == []:
                    ret.append(False)
                    break
                print(queue)
        return ret

solution = Solution()
print(solution.checkIfPrerequisite(5,
[[0,1],[1,2],[2,3],[3,4]],
[[0,4],[4,0],[1,3],[3,0]],
))