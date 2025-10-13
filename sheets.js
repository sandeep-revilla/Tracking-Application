/*******************************
  Raw-sheet triggered parser + immediate alerts + nightly 10 PM IST email
  - Install triggers with installTriggers()
  - Config in CONFIG block below
********************************/

/* ========== CONFIG ========== */
var RAW_SHEET_NAME = 'SMS received';        // sheet that contains incoming SMS/raw rows
var OWN_MASKS = ['*4219','*6516'];        // your masked accounts (or leave [] to auto-detect)
var IGNORE_NAME_REGEX = /\bsandeep\b/i;
var SUSPICIOUS_THRESHOLD = 5000;  // flagged as suspicious
var EMAIL_THRESHOLD = 500;        // immediate email for debits >= this
var RECIPIENT_EMAIL = '';         // default: Session.getActiveUser().getEmail()
var ALLOWED_BANKS = ['HDFC Bank','Indian Bank']; // only records matching these keywords
/* ============================ */

/* ---------- Utilities & parser ---------- */
function escapeHtml(text) {
  if (text === null || text === undefined) return '';
  return String(text)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function extractAllMasks(msg) {
  if (!msg) return [];
  var masks = []; var re = /\*\d{2,6}/g; var m;
  while ((m = re.exec(msg)) !== null) masks.push(m[0]);
  return Array.from(new Set(masks));
}

function extractFromToMasks(msg) {
  var fromMask = null, toMask = null;
  if (!msg) return { fromMask:null, toMask:null };
  var fromMatch = msg.match(/From[^*\n\r]*?(\*\d{2,6})/i);
  var toMatch = msg.match(/To[^*\n\r]*?(\*\d{2,6})/i);
  if (fromMatch) fromMask = fromMatch[1];
  if (toMatch) toMask = toMatch[1];
  var acMatch = msg.match(/(?:a\/c|A\/C|ac)\s*\*?(\d{2,6})/i);
  if (!fromMask && acMatch) fromMask = '*' + acMatch[1];
  var creditedMatch = msg.match(/credited to (?:a\/c )?\*?(\d{2,6})/i);
  if (creditedMatch) toMask = '*' + creditedMatch[1];
  var all = extractAllMasks(msg);
  if (!fromMask && all.length >= 1) fromMask = all[0];
  if (!toMask && all.length >= 2) toMask = all[1];
  return { fromMask: fromMask || null, toMask: toMask || null };
}

function parseTimestampStr(s, tz) {
  if (!s) return null;
  s = (''+s).trim();
  var t = s.replace(/\bat\b/gi,' ').replace(/([AP]M)$/i,' $1').replace(/\s+/g,' ');
  var d = new Date(t);
  if (!isNaN(d.getTime())) return d;
  var m = s.match(/(\d{1,2})\/(\d{1,2})\/(\d{4})/);
  var timeMatch = s.match(/(\d{1,2}):(\d{2})\s*(AM|PM)?/i);
  if (m) {
    var day = parseInt(m[1],10), month = parseInt(m[2],10)-1, year = parseInt(m[3],10);
    var hh = 0, mm = 0;
    if (timeMatch) {
      hh = parseInt(timeMatch[1],10); mm = parseInt(timeMatch[2],10);
      if (timeMatch[3]) {
        var ampm = (''+timeMatch[3]).toUpperCase();
        if (ampm === 'PM' && hh < 12) hh += 12;
        if (ampm === 'AM' && hh === 12) hh = 0;
      }
    }
    return new Date(year, month, day, hh, mm);
  }
  return null;
}

function extractAmountAndType(msg) {
  if (!msg) return { amount: null, type: null };
  var mAmt = msg.match(/(?:Rs\.?|INR)\s*([0-9,]+(?:\.\d+)?)/i);
  var amount = mAmt ? parseFloat(mAmt[1].replace(/,/g,'')) : null;
  var lower = (msg || '').toLowerCase();
  var type = null;
  if (/\bcredited\b/i.test(lower) || /\bcredit\b/.test(lower)) type = 'credit';
  else if (/\bsent\b/i.test(lower) && /\bfrom\b/i.test(lower)) type = 'debit';
  else if (/\bdebited\b|\bwithdrawn\b|\bpaid\b|\bpurchase\b|\bpurchased\b|\btransfer(ed)?\b/i.test(lower)) type = 'debit';
  if (/\bcredited\b/i.test(lower)) type = 'credit';
  return { amount: amount, type: type };
}

function extractBank(sender, msg) {
  var bankMap = {
    'hdfc': 'HDFC Bank',
    'indbank': 'Indian Bank',
    'indian bank': 'Indian Bank',
    'icici': 'ICICI Bank',
    'sbi': 'State Bank of India',
    'axis': 'Axis Bank',
    'kotak': 'Kotak Mahindra Bank',
    'yes': 'Yes Bank',
    'paytm': 'Paytm',
    'phonepe': 'PhonePe',
    'googlepay': 'Google Pay',
    'gpay': 'Google Pay',
    'upi': 'UPI'
  };
  if (sender) {
    var s = (''+sender).toLowerCase();
    for (var key in bankMap) if (s.indexOf(key) !== -1) return bankMap[key];
  }
  if (msg) {
    var m = (''+msg).toLowerCase();
    for (var key in bankMap) if (m.indexOf(key) !== -1) return bankMap[key];
    var bankNameMatch = msg.match(/([A-Za-z0-9 &]+?)\s+Bank/i);
    if (bankNameMatch) return bankNameMatch[1].trim() + ' Bank';
  }
  return sender || 'Unknown';
}

function buildAllowedKeywords() {
  var kw = ALLOWED_BANKS.map(function(b){ return (''+b).toLowerCase().replace(/bank/g,'').trim().split(/\s+/)[0]; });
  return Array.from(new Set(kw));
}

/* ---------- Deduplication helpers ----------- */

// normalize text for hashing
function normalizeForHash(s) {
  if (s === null || s === undefined) return '';
  return (''+s).replace(/\s+/g,' ').trim().toLowerCase();
}

// create a short hash key for a row (sender + normalized message + amount + minute-precision timestamp)
function computeMessageHash(sender, message, amount, dt, tz) {
  var s = normalizeForHash(sender);
  var m = normalizeForHash(message);
  var a = (amount === null || amount === undefined) ? '' : String(Math.round(Number(amount)*100)/100); // 2-dec precision
  var t = '';
  if (dt) {
    // minute precision avoids tiny second differences while still distinguishing different times
    t = Utilities.formatDate(dt, tz || SpreadsheetApp.getActive().getSpreadsheetTimeZone(), 'yyyy-MM-dd HH:mm');
  }
  return [s, m, a, t].join('||');
}

// read existing message hashes from History Transactions into a Set object for O(1) lookup
function loadExistingHashes() {
  var ss = SpreadsheetApp.getActive();
  var histSheet = ss.getSheetByName('History Transactions');
  var set = {};
  if (!histSheet) return set;
  var lastRow = histSheet.getLastRow();
  var lastCol = histSheet.getLastColumn();
  if (lastRow < 2) return set;
  var all = histSheet.getRange(1,1,lastRow,lastCol).getValues();
  var headers = all[0].map(function(h){ return (''+h).toLowerCase(); });
  var hashIdx = -1;
  for (var i=0;i<headers.length;i++) {
    if (headers[i].indexOf('messagehash') !== -1) { hashIdx = i; break; }
  }
  if (hashIdx === -1) return set;
  for (var r=1; r<all.length; r++) {
    var h = all[r][hashIdx];
    if (h) set[''+h] = true;
  }
  return set;
}

/* ---------- Core: process newly added rows in RAW sheet ---------- */
function processNewRawRows(sheet) {
  var ss = SpreadsheetApp.getActive();
  var tz = ss.getSpreadsheetTimeZone();
  var props = PropertiesService.getScriptProperties();
  var lastProcessed = parseInt(props.getProperty('RAW_LAST_PROCESSED_ROW') || '1', 10); // header row = 1
  var lastRow = sheet.getLastRow();
  var lastCol = sheet.getLastColumn();

  if (lastRow <= lastProcessed) {
    return; // nothing new
  }

  // --- load existing message hashes from History to dedupe ---
  var existingHashes = loadExistingHashes();

  // Ensure History Transactions has MessageHash column so we store new hashes.
  var histSheet = ss.getSheetByName('History Transactions');
  if (!histSheet) {
    histSheet = ss.insertSheet('History Transactions');
    var outHeader = ['DateTime','Bank','Type','Amount','Sender','FromMask','ToMask','Suspicious','IgnoreReason','Message','SourceRow','MessageHash'];
    histSheet.getRange(1,1,1,outHeader.length).setValues([outHeader]);
  } else {
    // ensure header contains MessageHash; add if missing
    var currentHeaders = histSheet.getRange(1,1,1,histSheet.getLastColumn()).getValues()[0];
    var hasHash = false;
    for (var i=0;i<currentHeaders.length;i++){
      if ((currentHeaders[i]||'').toString().toLowerCase().indexOf('messagehash') !== -1) { hasHash = true; break; }
    }
    if (!hasHash) {
      histSheet.getRange(1,currentHeaders.length+1).setValue('MessageHash');
    }
  }

  var allData = sheet.getRange(1,1,lastRow,lastCol).getValues();
  var headers = allData[0].map(function(h){ return (''+h).toLowerCase(); });
  function findHeaderIndex(possibleNames) {
    for (var i=0;i<headers.length;i++){
      for (var j=0;j<possibleNames.length;j++){
        if (headers[i].indexOf(possibleNames[j].toLowerCase()) !== -1) return i;
      }
    }
    return -1;
  }
  var idxTimestamp = findHeaderIndex(['timestamp','date','time','received','september']);
  var idxSender = findHeaderIndex(['sender','from','vm-','ad-','bt-']);
  var idxMessage = findHeaderIndex(['message','sms','text','body']);
  if (idxMessage === -1) idxMessage = allData[0].length - 1;
  if (idxTimestamp === -1) idxTimestamp = 0;

  var allowedKeywords = buildAllowedKeywords();

  var historyRows = [];
  var todayRowsToAppend = [];
  var suspiciousPendingRows = [];
  var today = new Date();

  var recipient = RECIPIENT_EMAIL && RECIPIENT_EMAIL.length>0 ? RECIPIENT_EMAIL : Session.getEffectiveUser().getEmail();
  if (!recipient) recipient = Session.getEffectiveUser().getEmail();

  for (var r = lastProcessed; r < lastRow; r++) {
    var row = allData[r];
    var rawTimestamp = row[idxTimestamp];
    var sender = row[idxSender];
    var message = row[idxMessage];
    var msgStr = (message === undefined || message === null) ? '' : String(message);

    var dt = parseTimestampStr(rawTimestamp, tz);
    if (!dt) {
      var mm = msgStr.match(/(\d{1,2}\/\d{1,2}\/\d{4})/);
      if (mm) dt = parseTimestampStr(mm[0], tz);
    }

    var at = extractAmountAndType(msgStr);
    if (!at || at.amount === null) {
      continue; // not a txn
    }

    var masks = extractFromToMasks(msgStr);
    var fromMask = masks.fromMask;
    var toMask = masks.toMask;

    var ignoreReason = '';
    if (IGNORE_NAME_REGEX.test(msgStr)) ignoreReason = 'Self-transfer by name';
    if (!ignoreReason && fromMask && toMask) {
      var ownSet = {}; OWN_MASKS.forEach(function(x){ ownSet[x]=true; });
      if ((ownSet[fromMask] && ownSet[toMask]) || (fromMask === toMask && ownSet[fromMask])) {
        ignoreReason = 'Self-transfer between own masked accounts';
      }
    }

    var bank = extractBank(sender, msgStr);
    var lowBank = (bank || '').toLowerCase();
    var keep = false;
    for (var k=0;k<allowedKeywords.length;k++){
      if (lowBank.indexOf(allowedKeywords[k]) !== -1) { keep = true; break; }
    }
    if (!keep) {
      // skip entirely
      continue;
    }

    var type = at.type || 'unknown';
    var amount = at.amount || 0;
    var suspicious = (amount >= SUSPICIOUS_THRESHOLD) ? 'Yes' : 'No';
    var dtFormatted = dt ? Utilities.formatDate(dt, tz, 'yyyy-MM-dd HH:mm:ss') : '';

    // compute message hash for dedupe
    var rowHash = computeMessageHash(sender, msgStr, amount, dt, tz);

    // skip duplicates (found in History)
    if (existingHashes[rowHash]) {
      Logger.log('Skipping duplicate row ' + (r+1) + ' (hash exists).');
      continue;
    }

    // append to History (store as string for readable history) + add hash
    historyRows.push([dtFormatted, bank, type, amount, sender||'', fromMask||'', toMask||'', suspicious, ignoreReason, msgStr, r+1, rowHash]);

    // mark locally to avoid duplicates within same run
    existingHashes[rowHash] = true;

    // immediate email for debit >= EMAIL_THRESHOLD (and not ignored)
    if (type === 'debit' && amount >= EMAIL_THRESHOLD && !ignoreReason) {
      try {
        var subject = 'Alert: Debit Transaction of Rs.' + amount.toFixed(2);
        var htmlBody = `<p>A debit transaction has been detected:</p>
          <table border="1" cellpadding="5" cellspacing="0">
            <tr><th>DateTime</th><th>Bank</th><th>Amount</th><th>FromMask</th><th>ToMask</th><th>Message</th></tr>
            <tr>
              <td>${escapeHtml(dtFormatted)}</td>
              <td>${escapeHtml(bank)}</td>
              <td>₹${amount.toFixed(2)}</td>
              <td>${escapeHtml(fromMask||'')}</td>
              <td>${escapeHtml(toMask||'')}</td>
              <td>${escapeHtml(msgStr).replace(/\n/g,'<br>')}</td>
            </tr>
          </table>`;
        MailApp.sendEmail({ to: recipient, subject: subject, htmlBody: htmlBody, body: 'Debit transaction: Rs.' + amount.toFixed(2) + ' on ' + dtFormatted });
      } catch (e) {
        Logger.log('Immediate email failed for row ' + (r+1) + ': ' + e);
      }
    }

    // buffer suspicious pending (for nightly mail) if suspicious and not ignored
    if (suspicious === 'Yes' && !ignoreReason) {
      suspiciousPendingRows.push([Utilities.formatDate(dt, tz, 'yyyy-MM-dd'), dtFormatted, bank, type, amount, sender||'', fromMask||'', toMask||'', msgStr, r+1]);
    }

    // add to today's table if txn date is today and not ignored
    if (dt && Utilities.formatDate(dt, tz, 'yyyy-MM-dd') === Utilities.formatDate(today, tz, 'yyyy-MM-dd') && !ignoreReason) {
      // store Date object in first column for consistent date handling
      todayRowsToAppend.push([dt, bank, type, amount, sender||'', fromMask||'', toMask||'', suspicious, '', msgStr, r+1]);
    }
  }

  // append history rows
  if (historyRows.length > 0) {
    // histSheet is already ensured above to exist and have MessageHash header
    histSheet.getRange(histSheet.getLastRow()+1,1,historyRows.length,historyRows[0].length).setValues(historyRows);
    histSheet.getRange(histSheet.getLastRow()-historyRows.length+1,4,historyRows.length,1).setNumberFormat('#,##0.00');
  }

  // append today rows to Today Transactions (first column is Date object)
  if (todayRowsToAppend.length > 0) {
    var todaySheet = ss.getSheetByName('Today Transactions');
    var outHeaderT = ['DateTime','Bank','Type','Amount','Sender','FromMask','ToMask','Suspicious','IgnoreReason','Message','SourceRow'];
    if (!todaySheet) {
      todaySheet = ss.insertSheet('Today Transactions');
      todaySheet.getRange(1,1,1,outHeaderT.length).setValues([outHeaderT]);
    }
    todaySheet.getRange(todaySheet.getLastRow()+1,1,todayRowsToAppend.length,todayRowsToAppend[0].length).setValues(todayRowsToAppend);
    // format first col as datetime and amount column as number
    todaySheet.getRange(todaySheet.getLastRow()-todayRowsToAppend.length+1,1,todayRowsToAppend.length,1).setNumberFormat('yyyy-MM-dd HH:mm:ss');
    todaySheet.getRange(todaySheet.getLastRow()-todayRowsToAppend.length+1,4,todayRowsToAppend.length,1).setNumberFormat('#,##0.00');
  }

  // append suspicious pending rows
  if (suspiciousPendingRows.length > 0) {
    var sp = ss.getSheetByName('Suspicious Pending');
    var spHeader = ['Date','DateTime','Bank','Type','Amount','Sender','FromMask','ToMask','Message','SourceRow'];
    if (!sp) {
      sp = ss.insertSheet('Suspicious Pending');
      sp.getRange(1,1,1,spHeader.length).setValues([spHeader]);
    }
    sp.getRange(sp.getLastRow()+1,1,suspiciousPendingRows.length,suspiciousPendingRows[0].length).setValues(suspiciousPendingRows);
    sp.getRange(sp.getLastRow()-suspiciousPendingRows.length+1,5,suspiciousPendingRows.length,1).setNumberFormat('#,##0.00');
  }

  // update the pointer so processed rows won't be processed again
  props.setProperty('RAW_LAST_PROCESSED_ROW', String(lastRow));
}

/* ---------- Update Today Summary (robust to Date objects & strings) ---------- */
function updateTodaySummary() {
  var ss = SpreadsheetApp.getActive();
  var tz = ss.getSpreadsheetTimeZone();
  var today = new Date();
  var todayStr = Utilities.formatDate(today, tz, 'yyyy-MM-dd');
  var todaySheet = ss.getSheetByName('Today Transactions');
  var totalDebit = 0, totalCredit = 0, countDebit = 0, countCredit = 0, suspiciousCount = 0;

  if (todaySheet) {
    var vals = todaySheet.getDataRange().getValues();
    if (vals && vals.length > 1) {
      for (var r = 1; r < vals.length; r++) {
        var row = vals[r];
        var dtCell = row[0];

        // Normalize dtCell to a yyyy-MM-dd string (works for Date objects and strings)
        var rowDateStr = '';
        if (dtCell instanceof Date) {
          rowDateStr = Utilities.formatDate(dtCell, tz, 'yyyy-MM-dd');
        } else if (dtCell !== null && dtCell !== undefined && String(dtCell).trim() !== '') {
          var s = String(dtCell).trim();
          var m = s.match(/(\d{4}-\d{2}-\d{2})/);
          if (m) rowDateStr = m[1];
          else {
            var tryDate = new Date(s);
            if (!isNaN(tryDate.getTime())) rowDateStr = Utilities.formatDate(tryDate, tz, 'yyyy-MM-dd');
            else rowDateStr = '';
          }
        } else {
          rowDateStr = '';
        }

        // Skip rows that do not belong to today's date
        if (rowDateStr !== todayStr) continue;

        var type = (row[2] || '').toString().toLowerCase();
        var amount = parseFloat(row[3]) || 0;
        var suspicious = (row[7] || 'No').toString().toLowerCase();

        if (type === 'debit') { totalDebit += amount; countDebit++; }
        else if (type === 'credit') { totalCredit += amount; countCredit++; }
        if (suspicious === 'yes') suspiciousCount++;
      }
    }
  }

  var sumSheet = ss.getSheetByName('Today Summary');
  if (!sumSheet) sumSheet = ss.insertSheet('Today Summary');
  sumSheet.clearContents();
  var rows = [
    ['Date', todayStr],
    ['Total Debit Amount', totalDebit],
    ['Total Credit Amount', totalCredit],
    ['Debit Count', countDebit],
    ['Credit Count', countCredit],
    ['Suspicious Count', suspiciousCount],
    ['Own Masks', OWN_MASKS.join(', ') || 'None'],
    ['Allowed Banks (keywords)', buildAllowedKeywords().join(', ')]
  ];
  sumSheet.getRange(1,1,rows.length,2).setValues(rows);
  if (rows.length >= 2) sumSheet.getRange(2,2,2,1).setNumberFormat('#,##0.00');
}

/* ---------- onEdit trigger handler (installable) ---------- */
function onEditHandler(e) {
  try {
    if (!e || !e.range) return;
    var sheet = e.range.getSheet();
    if (!sheet) return;
    if (sheet.getName() !== RAW_SHEET_NAME) return; // only respond to edits on Raw sheet

    processNewRawRows(sheet);
    // update summary after processing new rows
    updateTodaySummary();

  } catch (err) {
    Logger.log('onEditHandler error: ' + err);
  }
}

/* ---------- onChange trigger handler (catch programmatic appends like IFTTT) ---------- */
function onChangeHandler(e) {
  try {
    // If changeType exists, only proceed for INSERT_ROW or OTHER (defensive).
    if (e && e.changeType) {
      if (e.changeType !== 'INSERT_ROW' && e.changeType !== 'OTHER') return;
    }
    var ss = SpreadsheetApp.getActive();
    var sheet = ss.getSheetByName(RAW_SHEET_NAME);
    if (!sheet) return;

    // processNewRawRows checks RAW_LAST_PROCESSED_ROW and exits quickly if nothing new
    processNewRawRows(sheet);
    updateTodaySummary();
  } catch (err) {
    Logger.log('onChangeHandler error: ' + err);
  }
}

/* ---------- send daily email at 10 PM IST, then clear Today tables ---------- */
function sendDailyTransactionsEmailAt10PM() {
  var ss = SpreadsheetApp.getActive();
  var tz = ss.getSpreadsheetTimeZone();
  var recipient = RECIPIENT_EMAIL && RECIPIENT_EMAIL.length>0 ? RECIPIENT_EMAIL : Session.getEffectiveUser().getEmail();
  if (!recipient) {
    Logger.log('No recipient email available.');
    return;
  }
  var today = new Date();
  var todayStr = Utilities.formatDate(today, tz, 'yyyy-MM-dd');

  // ensure summary up-to-date
  updateTodaySummary();

  // get summary
  var sumSheet = ss.getSheetByName('Today Summary');
  var summaryLines = [];
  if (sumSheet) {
    var vals = sumSheet.getDataRange().getValues();
    for (var i=0;i<vals.length;i++){
      summaryLines.push('<tr><td><strong>' + escapeHtml(vals[i][0]||'') + '</strong></td><td>' + escapeHtml(String(vals[i][1]||'')) + '</td></tr>');
    }
  } else {
    summaryLines.push('<tr><td>No summary available</td><td></td></tr>');
  }

  // build today's transactions table (robust to Date objects)
  var todaySheet = ss.getSheetByName('Today Transactions');
  var tableHtml = '';
  if (todaySheet) {
    var vals = todaySheet.getDataRange().getValues();
    if (vals && vals.length > 1) {
      tableHtml += '<table border="1" cellpadding="5" cellspacing="0" style="border-collapse:collapse;">';
      tableHtml += '<tr>';
      for (var h=0; h<vals[0].length; h++) tableHtml += '<th>' + escapeHtml(vals[0][h]||'') + '</th>';
      tableHtml += '</tr>';
      for (var r=1; r<vals.length; r++) {
        var row = vals[r];
        var dtCell = row[0];
        var rowDateStr = '';
        if (dtCell instanceof Date) rowDateStr = Utilities.formatDate(dtCell, tz, 'yyyy-MM-dd');
        else {
          var s = String(dtCell||'').trim();
          var m = s.match(/(\d{4}-\d{2}-\d{2})/);
          if (m) rowDateStr = m[1];
          else {
            var tryDate = new Date(s);
            if (!isNaN(tryDate.getTime())) rowDateStr = Utilities.formatDate(tryDate, tz, 'yyyy-MM-dd');
            else rowDateStr = '';
          }
        }
        if (rowDateStr !== todayStr) continue;
        tableHtml += '<tr>';
        for (var c=0; c<row.length; c++) {
          var v = row[c];
          if (v instanceof Date) v = Utilities.formatDate(v, tz, 'yyyy-MM-dd HH:mm:ss');
          if (typeof v === 'number') v = '₹' + v.toFixed(2);
          tableHtml += '<td>' + escapeHtml(String(v||'')) + '</td>';
        }
        tableHtml += '</tr>';
      }
      tableHtml += '</table>';
    } else {
      tableHtml = '<p>No today transactions found.</p>';
    }
  } else {
    tableHtml = '<p>No Today Transactions sheet available.</p>';
  }

  // Suspicious Pending for today's date
  var sp = ss.getSheetByName('Suspicious Pending');
  var suspiciousHtml = '';
  if (sp) {
    var spVals = sp.getDataRange().getValues();
    var header = spVals[0] || [];
    var found = [];
    for (var r=1; r<spVals.length; r++){
      if ((spVals[r][0] || '') === todayStr) found.push(spVals[r]);
    }
    if (found.length > 0) {
      suspiciousHtml += '<p>Suspicious transactions added today:</p>';
      suspiciousHtml += '<table border="1" cellpadding="5" cellspacing="0" style="border-collapse:collapse;"><tr>';
      for (var hh=0; hh<header.length; hh++) suspiciousHtml += '<th>' + escapeHtml(header[hh]||'') + '</th>';
      suspiciousHtml += '</tr>';
      for (var i=0;i<found.length;i++){
        suspiciousHtml += '<tr>';
        for (var j=0; j<found[i].length; j++){
          var v = found[i][j];
          if (typeof v === 'number') v = '₹' + v.toFixed(2);
          suspiciousHtml += '<td>' + escapeHtml(String(v||'')) + '</td>';
        }
        suspiciousHtml += '</tr>';
      }
      suspiciousHtml += '</table>';
    } else {
      suspiciousHtml = '<p>No suspicious transactions were added today.</p>';
    }
  } else {
    suspiciousHtml = '<p>No Suspicious Pending sheet found.</p>';
  }

  // compose email
  var html = `<h3>Daily Transactions — ${todayStr}</h3>`;
  html += '<h4>Summary</h4>';
  html += '<table>' + summaryLines.join('') + '</table>';
  html += '<h4>All today\'s transactions</h4>' + tableHtml;
  html += '<h4>Suspicious transactions added today</h4>' + suspiciousHtml;

  try {
    MailApp.sendEmail({ to: recipient, subject: `Daily Transactions & Suspicious Report — ${todayStr}`, htmlBody: html, body: `Daily Transactions for ${todayStr}` });
  } catch (e) {
    Logger.log('sendDailyTransactionsEmailAt10PM mail error: ' + e);
  }

  // After sending, clear Today's sheets (reset them so next day starts fresh)
  if (todaySheet) {
    var outHeaderT = ['DateTime','Bank','Type','Amount','Sender','FromMask','ToMask','Suspicious','IgnoreReason','Message','SourceRow'];
    todaySheet.clearContents();
    todaySheet.getRange(1,1,1,outHeaderT.length).setValues([outHeaderT]);
  }
  if (sumSheet) {
    sumSheet.clearContents();
  }

  // Remove today's entries from Suspicious Pending so they are not repeated
  if (sp) {
    var spVals = sp.getDataRange().getValues();
    var rowsToKeep = [spVals[0]]; // keep header
    for (var r=1; r<spVals.length; r++){
      if ((spVals[r][0] || '') !== todayStr) rowsToKeep.push(spVals[r]);
    }
    sp.clearContents();
    if (rowsToKeep.length > 0) sp.getRange(1,1,rowsToKeep.length,rowsToKeep[0].length).setValues(rowsToKeep);
  }
}

/* ---------- Installation helpers (create triggers) ---------- */
function installTriggers() {
  // delete any old triggers we manage
  deleteTriggersForHandler('onEditHandler');
  deleteTriggersForHandler('onChangeHandler');
  deleteTriggersForHandler('sendDailyTransactionsEmailAt10PM');

  // installable onEdit (keeps manual edits responsive)
  ScriptApp.newTrigger('onEditHandler').forSpreadsheet(SpreadsheetApp.getActive()).onEdit().create();

  // install onChange to catch API/IFTTT programmatic row inserts
  ScriptApp.newTrigger('onChangeHandler').forSpreadsheet(SpreadsheetApp.getActive()).onChange().create();

  // daily 10 PM trigger (low frequency)
  ScriptApp.newTrigger('sendDailyTransactionsEmailAt10PM').timeBased().atHour(22).everyDays(1).create();

  SpreadsheetApp.getUi().alert('Triggers installed: onEdit (for manual edits), onChange (for programmatic appends like IFTTT), and daily 22:00 email. Set project timezone to Asia/Kolkata for 10 PM IST.');
}

function uninstallTriggers() {
  deleteTriggersForHandler('onEditHandler');
  deleteTriggersForHandler('onChangeHandler');
  deleteTriggersForHandler('sendDailyTransactionsEmailAt10PM');
  SpreadsheetApp.getUi().alert('Triggers removed.');
}

function deleteTriggersForHandler(fn) {
  var all = ScriptApp.getProjectTriggers();
  for (var i=0;i<all.length;i++){
    if (all[i].getHandlerFunction() === fn) ScriptApp.deleteTrigger(all[i]);
  }
}

/* ---------- Helper to initialize pointer ---------- */
function initRawPointer() {
  var ss = SpreadsheetApp.getActive();
  var sheet = ss.getSheetByName(RAW_SHEET_NAME);
  if (!sheet) { SpreadsheetApp.getUi().alert('Raw sheet not found.'); return; }
  var props = PropertiesService.getScriptProperties();
  props.setProperty('RAW_LAST_PROCESSED_ROW', String(sheet.getLastRow()));
  SpreadsheetApp.getUi().alert('Initialized RAW_LAST_PROCESSED_ROW to ' + sheet.getLastRow());
}
