clear; clc;

%% --- Initialization ---
x   = -4:0.1:4;    % Temperature (°C)
y   = 0:0.1:2;     % Mass (kg)
w   = 600:5:1200;  % Power (Watt)
z   = 0:0.1:10;    % Duration (min)

nx  = length(x);
ny  = length(y);
nw  = length(w);
nz  = length(z);

%% --- Output Memberships (Power & Time) ---
[lowW, midW, highW] = deal(zeros(1, nw));
[shortZ, medZ, longZ] = deal(zeros(1, nz));

for i = 1:nw
    currW = w(i);
    lowW(i)  = max(0, (800 - currW)/200 * (currW >= 600 && currW <= 800));
    midW(i)  = max(0, ((currW - 600)/300 * (currW >= 600 && currW <= 900)) + ((1200 - currW)/300 * (currW > 900 && currW <= 1200)));
    highW(i) = max(0, (currW - 1000)/200 * (currW >= 1000 && currW <= 1200));
end

for j = 1:nz
    currZ = z(j);
    shortZ(j) = max(0, (5 - currZ)/5 * (currZ >= 0 && currZ <= 5));
    medZ(j)   = max(0, (currZ/5 * (currZ >= 0 && currZ <= 5)) + ((10 - currZ)/5 * (currZ > 5 && currZ <= 10)));
    longZ(j)  = max(0, (currZ - 5)/5 * (currZ >= 5 && currZ <= 10));
end

%% --- Input Memberships (x = Temp, y = Mass) ---
coldX = zeros(1, nx);
mildX = zeros(1, nx);
hotX  = zeros(1, nx);

for i = 1:nx
    valX = x(i);
    coldX(i) = (valX >= -4 && valX <= 0) * (-valX / 4);
    if valX >= -3 && valX <= 0
        mildX(i) = (valX + 3) / 3;
    elseif valX > 0 && valX <= 3
        mildX(i) = (3 - valX) / 3;
    else
        mildX(i) = 0;
    end
    hotX(i) = (valX >= 0 && valX <= 4) * (valX / 4);
end

lightY = zeros(1, ny);
medY   = zeros(1, ny);
heavyY = zeros(1, ny);

for j = 1:ny
    valY = y(j);
    lightY(j) = (valY >= 0 && valY <= 1) * (1 - valY);
    if valY >= 0 && valY <= 1
        medY(j) = valY;
    elseif valY > 1 && valY <= 2
        medY(j) = 2 - valY;
    else
        medY(j) = 0;
    end
    heavyY(j) = (valY >= 1 && valY <= 2) * (valY - 1);
end

%% --- Rule Activation ---
totalPairs = nx * ny;
[alphaLowW, alphaMedW, alphaHighW] = deal(zeros(1, totalPairs));
[alphaShortZ, alphaMedZ, alphaLongZ] = deal(zeros(1, totalPairs));

idx = 1;
for i = 1:nx
    for j = 1:ny
        r1 = min(coldX(i), heavyY(j));
        r4 = min(coldX(i), medY(j));
        r7 = min(coldX(i), lightY(j));
        alphaHighW(idx) = max([r1, r4, r7]);

        r2 = min(mildX(i), heavyY(j));
        r5 = min(mildX(i), medY(j));
        r8 = min(mildX(i), lightY(j));
        alphaMedW(idx) = max([r2, r5, r8]);

        r3 = min(hotX(i), heavyY(j));
        r6 = min(hotX(i), medY(j));
        r9 = min(hotX(i), lightY(j));
        alphaLowW(idx) = max([r3, r6, r9]);

        rz1 = min(coldX(i), heavyY(j));
        rz4 = min(mildX(i), heavyY(j));
        rz7 = min(hotX(i), heavyY(j));
        alphaLongZ(idx) = max([rz1, rz4, rz7]);

        rz2 = min(coldX(i), medY(j));
        rz5 = min(mildX(i), medY(j));
        rz8 = min(hotX(i), medY(j));
        alphaMedZ(idx) = max([rz2, rz5, rz8]);

        rz3 = min(coldX(i), lightY(j));
        rz6 = min(mildX(i), lightY(j));
        rz9 = min(hotX(i), lightY(j));
        alphaShortZ(idx) = max([rz3, rz6, rz9]);

        idx = idx + 1;
    end
end

%% --- Aggregation ---
clipLowW  = zeros(totalPairs, nw);
clipMedW  = zeros(totalPairs, nw);
clipHighW = zeros(totalPairs, nw);

clipShortZ = zeros(totalPairs, nz);
clipMedZ   = zeros(totalPairs, nz);
clipLongZ  = zeros(totalPairs, nz);

for i = 1:totalPairs
    for j = 1:nw
        clipLowW(i,j)  = min(lowW(j),  alphaLowW(i));
        clipMedW(i,j)  = min(midW(j),  alphaMedW(i));
        clipHighW(i,j) = min(highW(j), alphaHighW(i));
    end
    for j = 1:nz
        clipShortZ(i,j) = min(shortZ(j), alphaShortZ(i));
        clipMedZ(i,j)   = min(medZ(j),   alphaMedZ(i));
        clipLongZ(i,j)  = min(longZ(j),  alphaLongZ(i));
    end
end

aggW = max(max(clipLowW, clipMedW), clipHighW);
aggZ = max(max(clipShortZ, clipMedZ), clipLongZ);

%% --- Defuzzification + Plots ---
[xGrid, yGrid] = meshgrid(y, x);
[COG_W, COG_Z, MOM_W, MOM_Z, mMOM_W, mMOM_Z, CA_W, CA_Z] = deal(zeros(nx, ny));

idx = 1;
for i = 1:nx
    for j = 1:ny
        wp = aggW(idx,:);
        zp = aggZ(idx,:);

        COG_W(i,j) = sum(wp .* w) / (sum(wp) + eps);
        COG_Z(i,j) = sum(zp .* z) / (sum(zp) + eps);

        maxW = max(wp); maxZ = max(zp);
        MOM_W(i,j) = mean(w(abs(wp - maxW) < 1e-6));
        MOM_Z(i,j) = mean(z(abs(zp - maxZ) < 1e-6));

        iW = find(abs(wp - maxW) < 1e-6);
        iZ = find(abs(zp - maxZ) < 1e-6);
        mMOM_W(i,j) = (min(w(iW)) + max(w(iW))) / 2;
        mMOM_Z(i,j) = (min(z(iZ)) + max(z(iZ))) / 2;

        aW = [alphaLowW(idx), alphaMedW(idx), alphaHighW(idx)];
        aZ = [alphaShortZ(idx), alphaMedZ(idx), alphaLongZ(idx)];

        repW = [median(w(abs(wp - aW(1)) < 1e-6)), median(w(abs(wp - aW(2)) < 1e-6)), median(w(abs(wp - aW(3)) < 1e-6))];
        repZ = [median(z(abs(zp - aZ(1)) < 1e-6)), median(z(abs(zp - aZ(2)) < 1e-6)), median(z(abs(zp - aZ(3)) < 1e-6))];

        repW(isnan(repW)) = 0; repZ(isnan(repZ)) = 0;
        CA_W(i,j) = sum(aW .* repW) / (sum(aW) + eps);
        CA_Z(i,j) = sum(aZ .* repZ) / (sum(aZ) + eps);

        idx = idx + 1;
    end
end

figure; surf(xGrid, yGrid, COG_W); title('Watt(COG)'); xlabel('Temp'); ylabel('Weight'); zlabel('Watt');
figure; surf(xGrid, yGrid, COG_Z); title('Time(COG)');  xlabel('Temp'); ylabel('Weight'); zlabel('Time');

figure; surf(xGrid, yGrid, MOM_W); title('Watt (MOM)'); xlabel('Temp'); ylabel('Weight'); zlabel('Watt');
figure; surf(xGrid, yGrid, MOM_Z); title('Time (MOM)');  xlabel('Temp'); ylabel('Weight'); zlabel('Time');

figure; surf(xGrid, yGrid, mMOM_W); title('Watt (Modified MOM)'); xlabel('Temp'); ylabel('Weight'); zlabel('Watt');
figure; surf(xGrid, yGrid, mMOM_Z); title('Time (Modified MOM)'); xlabel('Temp'); ylabel('Weight'); zlabel('Time');

figure; surf(xGrid, yGrid, CA_W); title('Watt (CA)'); xlabel('Temp'); ylabel('Weight'); zlabel('Watt');
figure; surf(xGrid, yGrid, CA_Z); title('Time (CA)'); xlabel('Temp'); ylabel('Weight'); zlabel('Time');


%% --- 匯出結果為 Excel 表格 ---

% 建立 x-y 組合（x = 溫度，y = 重量）
x_label   = repmat(x', 1, ny);     % 溫度矩陣（橫軸）
y_label   = repmat(y, nx, 1);      % 重量矩陣（縱軸）

% 展平成向量
x_vec   = x_label(:);              % 所有溫度對應的值
y_vec   = y_label(:);              % 所有重量對應的值

% 找出 x 為整數，y 為 0.2 的倍數且 0 < y <= 2
isTarget = mod(x_vec, 1) == 0 & abs(mod(y_vec, 0.2)) < 1e-6 & y_vec > 0 & y_vec <= 2;
idx_all = find(isTarget);

% 篩選前 90 筆資料
idx = idx_all(1:min(90, numel(idx_all)));

% 取得唯一的溫度與重量值（作為欄與列）
intTemps = unique(x_vec(idx));
selMass  = unique(y_vec(idx));

% 輸出組合：檔名、結果向量、表格標題
outputs = {
    'COG_Time_Table.xlsx',     COG_Z(:),     'Time COG';
    'COG_Power_Table.xlsx',    COG_W(:),     'Power COG';
    'MOM_Time_Table.xlsx',     MOM_Z(:),     'Time MOM';
    'MOM_Power_Table.xlsx',    MOM_W(:),     'Power MOM';
    'ModifiedMOM_Time_Table.xlsx', mMOM_Z(:), 'Time Modified MOM';
    'ModifiedMOM_Power_Table.xlsx', mMOM_W(:), 'Power Modified MOM';
    'CA_Time_Table.xlsx',      CA_Z(:),      'Time CA';
    'CA_Power_Table.xlsx',     CA_W(:),      'Power CA'
};

for i = 1:size(outputs,1)
    filename   = outputs{i,1};     % 檔案名稱
    result     = outputs{i,2};     % 對應的數值結果（展平成向量）
    titleLabel = outputs{i,3};     % 標籤（未實際使用）

    % 初始化數值表格
    tableData = nan(numel(selMass), numel(intTemps));

    % 將選中索引對應資料填入表格
    for k = 1:length(idx)
        tval = x_vec(idx(k));
        wval = y_vec(idx(k));
        row = find(abs(selMass - wval) < 1e-6);
        col = find(abs(intTemps - tval) < 1e-6);
        tableData(row, col) = result(idx(k));
    end

    % 橫軸為溫度（欄位標題）
    colHeaders = arrayfun(@(x) sprintf('%g°C', x), intTemps, 'UniformOutput', false);
    % 縱軸為重量（列標題）
    rowHeaders = reshape(arrayfun(@(y) sprintf('%gkg', y), selMass, 'UniformOutput', false), [], 1);

    % 數值轉為三位小數字串格式
    formattedData = arrayfun(@(v) sprintf('%.3f', v), tableData, 'UniformOutput', false);

    % 整理為 cell array 格式
    headerRow  = [{' '}, reshape(colHeaders, 1, [])];
    dataRows   = [rowHeaders, reshape(formattedData, size(tableData))];
    outputCell = [headerRow; dataRows];

    % 寫入 Excel 檔案
    writecell(outputCell, filename);
end