from .utils import *
# import runnable.qr_code as qr_code
import runnable.qr_code as qr_code
from queue import Queue
from sklearn.cluster import KMeans


#
# API to detect QR code in images.
#
# Parameters:
#   img: (m * n * 3) numpy ndarray of RGB format
#       The input image to detect.
#
# Returns:
#   flag: boolean
#       Whether the QR code is detected.
#   center: tuple(int, int)
#       The center coordinate of the QR code.
#
def get_qr_code(img):
    return qr_code.get_qr_code(img)


#
# API to detect cylinder im images.
#
# Parameters:
#   img: (m * n * 3) numpy ndarray of RGB format
#       The input image to detect.
#
# Returns:
#   flag: boolean
#       Whether the cylinder is detected.
#   pos: tuple(int, int)
#       The position of the cylinder.
#
def get_cylinder(img):
    if img.shape[0] < 20 or img.shape[1] < 20:
        return False, (0, 0)
    img2 = np.zeros((img.shape[0], img.shape[1]))
    t1 = img[:, :, 0] / 2 - img[:, :, 1]
    t2 = img[:, :, 0] / 2 - img[:, :, 2]
    img2[(t1 > 0) * (t2 > 0) * (img[:, :, 0] > 150)] = 255
    # show_image(img2, gray=True)
    aa = img2.sum(axis=1)
    threshold = aa.max() * 0.8
    l = 0
    while aa[l] < threshold:
        l += 1
    r = aa.shape[0] - 1
    while aa[r] < threshold:
        r -= 1
    x = int((l + r) / 2)
    aa = img2.sum(axis=0)
    threshold = aa.max() * 0.8
    l = 0
    while aa[l] < threshold:
        l += 1
    r = aa.shape[0] - 1
    while aa[r] < threshold:
        r -= 1
    y = int((l + r) / 2)
    return True, (y, x)


#
# API to detect target(T) and end(E) in images.
#
# Parameters:
#   img: (m * n * 3) numpy ndarray of RGB format
#       The input image to detect.
#
# Returns:
#   flag: boolean
#       Whether E or T is detected.
#   results: tuple(tuple(int, int, int, int), str)
#       The first tuple is (min_x, min_y, max_x, max_y).
#       The letter may be '?', which means it's hard to determine whether it is
#       'T' or 'E'.
#
def get_E_or_T(img):
    gray_img = get_gray(img)
    _, th1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    tmp = cv2.blur(th1, (3, 3))
    contours, hierarchy = cv2.findContours(tmp.copy(), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    num = len(contours)
    hh = hierarchy[0]
    that = []
    for i in range(num):
        if hh[i][2] == -1:
            continue
        j = hh[i][2]
        if hh[j][0] != -1 or hh[j][1] != -1:
            continue
        if hh[j][2] == -1:
            continue
        j = hh[j][2]
        if hh[j][0] != -1 or hh[j][1] != -1 or hh[j][2] != -1:
            continue
        that.append(i)
    if len(that) < 0:
        return False, ((0, 0, 0, 0), '?')
    cc = contours[that[0]]
    min_x = cc.min(axis=0)[0][0]
    min_y = cc.min(axis=0)[0][1]
    max_x = cc.max(axis=0)[0][0]
    max_y = cc.max(axis=0)[0][1]
    new_img = tmp[min_y: max_y + 1, min_x: max_x + 1]
    m, n = new_img.shape
    letter = new_img[m // 5: m - m // 5, n // 5: n - n // 5]
    letter2 = rotate(letter, 180)
    myletter = np.int64(letter) * np.int64(letter2)
    myletter[myletter > 0] = 255
    myletter = np.uint8(myletter)
    cc2, hh2 = cv2.findContours(myletter.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_TC89_L1)
    ans = '?'
    if len(cc2) == 2:
        ans = 'T'
    elif len(cc2) == 4:
        ans = 'E'
    return True, ((min_x, min_y, max_x, max_y), ans)


#
# API to detect target(T) and end(E) in images.
#
# Parameters:
#   img: (m * n * 3) numpy ndarray of RGB format
#       The input image to detect.
#   eye: boolean
#       eye=True: left eye.
#       eye=False: right eye.
#
# Returns:
#   people: list of tuple(int, int)
#       Each tuple represent a coordinate of people.
#
def get_people(img, fuck_threshold):
    img2 = img.copy()
    img2 = rotate(img2, 180).copy()
    # show_image(img2)
    img2[(img2[:, :, 0] == 112) * (img2[:, :, 1] == 160) * (img2[:, :, 2] == 202)] = [255, 255, 255]
    img2 = cv2.resize(img2.copy(), (320, 180))
    visit = np.zeros((img2.shape[0], img2.shape[1]))
    belong = np.zeros(visit.shape)
    cnt = 0
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if not visit[i, j]:
                Q = Queue()
                Q.put((i, j))
                visit[i, j] = 1
                cnt += 1
                belong[i, j] = cnt
                dx = [-1, 0, 0, 1]
                dy = [0, -1, 1, 0]
                while not Q.empty():
                    x, y = Q.get()
                    for k in range(4):
                        xx = x + dx[k]
                        yy = y + dy[k]
                        if xx < 0 or xx >= img2.shape[0] or yy < 0 or yy >= \
                                img2.shape[1]:
                            continue
                        if visit[xx, yy] == 1:
                            continue
                        fuck1 = img2[x, y].astype(np.float64)
                        fuck2 = img2[xx, yy].astype(np.float64)
                        dist = np.abs(fuck1 - fuck2).sum()
                        if dist > 5:
                            continue
                        visit[xx, yy] = 1
                        belong[xx, yy] = cnt
                        Q.put((xx, yy))
    points = [[] for i in range(cnt + 1)]
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if belong[i, j] > 0:
                points[int(belong[i, j])].append((i, j))
    for pp in points:
        if len(pp) == 0:
            continue
        x = np.array([t[0] for t in pp])
        y = np.array([t[1] for t in pp])
        x0 = x.max()
        if x0 < 40:
            cc = 1
        elif x0 < 80:
            cc = 1.3
        elif x0 < 120:
            cc = 1.6
        elif x0 < 160:
            cc = 1.9
        else:
            cc = 2.2
        if (x.max() - x.min()) > 60 * cc or (y.max() - y.min()) > 60 * cc or len(pp) > 800 * cc * cc:
            for p in pp:
                img2[p[0], p[1]] = [255, 255, 255]
    # show_image(img2)
    fuck1 = np.array([172.87, 182.88, 188.63])
    fuck2 = np.array([189.49, 198.48, 204.345])
    fuck1_all = np.array([188.63, 182.88, 172.87])
    fuck2_all = np.array([204.345, 198.48, 189.49])
    tmp = (img2 < 2).sum(axis=2)
    img2[tmp == 3] = [255, 255, 255]
    img2[np.abs(img2 - fuck1_all).max(axis=2) < 20] = [255, 255, 255]
    img2[np.abs(img2 - fuck2_all).max(axis=2) < 20] = [255, 255, 255]
    img2 = cv2.resize(img2, (320, 180))
    # show_image(img2)
    visit = np.zeros((img2.shape[0], img2.shape[1]))
    belong = np.zeros(visit.shape)
    cnt = 0
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if not visit[i, j] and (img2[i, j] == 255).sum() < 3:
                Q = Queue()
                Q.put((i, j))
                visit[i, j] = 1
                cnt += 1
                belong[i, j] = cnt
                dx = [-1, 0, 0, 1]
                dy = [0, -1, 1, 0]
                while not Q.empty():
                    x, y = Q.get()
                    for k in range(4):
                        xx = x + dx[k]
                        yy = y + dy[k]
                        if xx < 0 or xx >= img2.shape[0] or yy < 0 or yy >= \
                                img2.shape[1]:
                            continue
                        if visit[xx, yy] == 1 or (
                                img2[xx, yy] == 255).sum() == 3:
                            continue
                        visit[xx, yy] = 1
                        belong[xx, yy] = cnt
                        Q.put((xx, yy))
    points = [[] for i in range(cnt + 1)]
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if belong[i, j] > 0:
                points[int(belong[i, j])].append((i, j))
    img3 = img2.copy()
    img3[:, :] = [255, 255, 255]
    threshold = fuck_threshold
    new_points = []
    for pp in points:
        if len(pp) == 0:
            continue
        x = np.array([t[0] for t in pp])
        y = np.array([t[1] for t in pp])
        x0 = x.max()
        if x0 < 40:
            cc = 1
        elif x0 < 80:
            cc = 1.3
        elif x0 < 120:
            cc = 1.6
        elif x0 < 160:
            cc = 1.9
        else:
            cc = 2.2
        if len(pp) > threshold * cc:
            xl = np.percentile(x, 20)
            xr = np.percentile(x, 80)
            yl = np.percentile(y, 20)
            yr = np.percentile(y, 80)
            # print(x.max(), x.min(), y.max(), y.min(), xr, xl, yr, yl, ' ', x.mean(), y.mean(), len(pp))
            if xr - xl < 1.3 * (yr - yl):
                continue
            new_points.append(pp)
            for p in pp:
                img3[p[0], p[1]] = img2[p[0], p[1]]
    # img3[(img3.sum(axis=2) < 200)] = [0, 255, 0]
    # show_image(img3)
    ans = []
    for pp in new_points:
        X = np.zeros((len(pp), 3))
        for i, p in enumerate(pp):
            color = img3[p[0], p[1]]
            X[i] = color
        # print(X)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
        # print(kmeans.cluster_centers_)
        # print(kmeans.labels_)
        num = np.zeros(3)
        for label in kmeans.labels_:
            num[label] += 1
        t = num.argmax()
        # for i, label in enumerate(kmeans.labels_):
        #     if label == t:
        #         print(X[i])
        color = kmeans.cluster_centers_[t]
        # tmp = []
        # for p in pp:
        #     if img3[p[0], p[1]].sum() < 200:
        #         tmp.append(p)
        # tmp = sorted(tmp)
        # if len(tmp) == 0:
        #     continue
        # x, y = tmp[0]
        y = np.array([t[1] for t in pp]).mean()
        x = np.array([t[0] for t in pp]).mean()
        xx = int(img.shape[0] - (x * 4 + 2))
        yy = int(img.shape[1] - (y * 4 + 2))
        ans.append(((yy, xx), color))
    return ans


if __name__ == '__main__':
    img = read_image('testcase/zuoyan.jpeg')
    print(get_people(img, True))
