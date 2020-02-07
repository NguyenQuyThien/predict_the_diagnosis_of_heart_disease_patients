def run(filePath, totalBenhNhan, result):
    """
    xử lý data, tổng hợp từ 4 file (chạy từng file 1)
    :param filePath:
    :param totalBenhNhan: lưu tổng số bệnh nhân đọc được đến lúc này
    :param result: list lưu data
    :return:
    """
    f = open("./first-data/{}".format( filePath ), "r")
    flagStop = 1
    while flagStop:
        duLieu1BenhNhan = ''
        for i in range(10):
            try:
                line = f.readline()
            except UnicodeDecodeError:
                continue
            # check eof of file and break while loop
            if (len(line) == 0):
                flagStop = 0
                break
            line2 = line.strip()
            if (i != 9):
                line2 += " "
            duLieu1BenhNhan += line2
        if (flagStop == 0):
            break

        duLieu1BenhNhan = duLieu1BenhNhan.strip()
        # hien thi du lieu 1 benh nhan tren 1 dong
        # print(duLieu1BenhNhan)

        # chuyen du lieu thanh mang
        mang1BenhNhan = duLieu1BenhNhan.split(' ')
        # print(mang1BenhNhan)
        result.append(mang1BenhNhan)
        # print( 'kich thuoc mang 1 benh nhan {}\n'.format( len(mang1BenhNhan) ))
        totalBenhNhan += 1
    f.close()
    return totalBenhNhan

if __name__ == "__main__":
    totalBenhNhan = 0
    result = []
    totalBenhNhan = run('hungarian.data', totalBenhNhan, result)
    totalBenhNhan = run('switzerland.data', totalBenhNhan, result)
    totalBenhNhan = run('long-beach-va.data', totalBenhNhan, result)
    totalBenhNhan = run('cleveland.data', totalBenhNhan, result)
    # print(totalBenhNhan)
    # bo 2 ban ghi cuoi trong file cleveland (data kys tuwj laj)
    result.pop()
    result.pop()
    # print(len(result))
    """
        result is writed to processed_file.csv
    """


# def save_data_to_processed_file(nameOfFile):
    #     with open('{}.csv'.format( nameOfFile ), mode='w') as processed_file:
    #         data_writer = csv.writer(processed_file, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
    #         data_writer.writerow(['Age', 'Sex', 'Cp', 'Trestbps', 'Chol', 'Fbs', 'Restecg', 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'Ca', 'Thal', 'Result'])
    #         for i in range( 719 , len(result) ):
    #             x = [result[i][2], result[i][3], result[i][8], result[i][9], result[i][11], result[i][15], result[i][18], result[i][31], result[i][37], result[i][39], result[i][40], result[i][43], result[i][50], result[i][57]]
    #             print(", ".join(x))
    #             data_writer.writerow(x)