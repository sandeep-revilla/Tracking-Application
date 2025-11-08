/***************************************
 Bulk RAW processing (batch safe) + nightly purge
 - Hourly scheduled sweep processes ALL rows in RAW (no deletes during day)
 - Nightly final sweep + purge RAW (clear content except header) at 23:00-23:59 project TZ
 - Idempotent via MessageHash in History Transactions
***************************************/

/* ========== CONFIG ========== */
var RAW_SHEET_NAME = 'SMS received';
var SENDER_REGEX = /\b(hdfc|hdfcbk|hdfc[\-\s]*bank|indb|indbnk|indbank|indian[\-\s]*bank)\b/i;
var OWN_MASKS = ['*4219','*6516'];        // your masked accounts (or [] to auto-detect)
var IGNORE_NAME_REGEX = /\bsandeep\b/i;
var SUSPICIOUS_THRESHOLD = 5000;
var ALLOWED_BANKS = ['HDFC Bank','Indian Bank'];
var BATCH_SIZE = 1200;                    // rows processed per batch - tune for your runtime
var PROCESS_TIME_BUFFER_SECONDS = 30;     // stop processing before Apps Script hard timeout
/* ============================ */

/* ---------------- Utilities ---------------- */
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
  var fromMatch = msg.match(/from[^*\n\r]*?(\*\d{2,6})/i);
  var toMatch = msg.match(/to[^*\n\r]*?(\*\d{2,6})/i);
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

/* Robust date parser - supports dd/mm/yyyy, dd-mm-yyyy, dd Mon yyyy, and common time formats */
function parseTimestampStr(s, tz) {
  if (!s) return null;
  s = (''+s).trim();
  // Normalize common separators
  s = s.replace(/\bat\b/gi, ' ').replace(/\s+/g,' ').trim();
  // Try direct Date
  var d = new Date(s);
  if (!isNaN(d.getTime())) return d;
  // dd/mm/yyyy or dd-mm-yyyy
  var m = s.match(/(\d{1,2})[\/-](\d{1,2})[\/-](\d{4})(?:.*?(\d{1,2}):(\d{2})(?:\s*(AM|PM))?)?/i);
  if (m) {
    var day = parseInt(m[1],10), month = parseInt(m[2],10)-1, year = parseInt(m[3],10);
    var hh = 0, mm = 0;
    if (m[4]) {
      hh = parseInt(m[4],10); mm = parseInt(m[5],10) || 0;
      if (m[6]) {
        var ampm = (''+m[6]).toUpperCase();
        if (ampm === 'PM' && hh < 12) hh += 12;
        if (ampm === 'AM' && hh === 12) hh = 0;
      }
    }
    return new Date(year, month, day, hh, mm);
  }
  // dd Mon yyyy
  var m2 = s.match(/(\d{1,2})[-\s]([A-Za-z]{3,9})[-\s](\d{4})(?:.*?(\d{1,2}):(\d{2})(?:\s*(AM|PM))?)?/i);
  if (m2) {
    var day2 = parseInt(m2[1],10);
    var monname = m2[2].toLowerCase();
    var monthMap = { jan:0,feb:1,mar:2,apr:3,may:4,jun:5,jul:6,aug:7,sep:8,oct:9,nov:10,dec:11};
    var month2 = monthMap[monname.substring(0,3)] || 0;
    var year2 = parseInt(m2[3],10);
    var hh2 = 0, mm2 = 0;
    if (m2[4]) {
      hh2 = parseInt(m2[4],10); mm2 = parseInt(m2[5],10) || 0;
      if (m2[6]) {
        var ampm2 = (''+m2[6]).toUpperCase();
        if (ampm2 === 'PM' && hh2 < 12) hh2 += 12;
        if (ampm2 === 'AM' && hh2 === 12) hh2 = 0;
      }
    }
    return new Date(year2, month2, day2, hh2, mm2);
  }
  return null;
}

/* ---------------- OTP / informational detection ---------------- */
function isOtpMessage(msg) {
  if (!msg) return false;
  var s = ('' + msg).toLowerCase();
  if (/\botp\b/.test(s) && /\b\d{3,7}\b/.test(s)) return true;
  if (/\bvalid till\b/.test(s)) return true;
  if (/\bdo not share otp\b/.test(s)) return true;
  if (/\bone-?time[- ]?password\b/.test(s)) return true;
  if (/\botp for (txn|transaction)\b/.test(s)) return true;
  return false;
}

function isInformationalAlert(msg) {
  if (!msg) return false;
  var s = ('' + msg).toLowerCase();
  var phrases = [ 'new limit', 'limit set', 'limit changed', 'limit updated', 'limit enabled', 'online domestic transactions limit', 'online transaction limit', 'card limit', 'daily limit', 'monthly limit', 'spending limit', 'limit:', 'limit -', 'limit –' ];
  for (var i = 0; i < phrases.length; i++) {
    if (s.indexOf(phrases[i]) !== -1) return true;
  }
  if (s.indexOf('limit') !== -1 && (s.indexOf('card') !== -1 || s.indexOf('transaction') !== -1 || s.indexOf('transactions') !== -1 || s.indexOf('enabled') !== -1 || s.indexOf('online') !== -1)) {
    return true;
  }
  if ((s.indexOf('not you?') !== -1 || s.indexOf('not you') !== -1 || s.indexOf('chat with us') !== -1 || s.indexOf('for assistance') !== -1) &&
      (s.indexOf('limit') !== -1 || s.indexOf('enabled') !== -1 || s.indexOf('settings') !== -1)) {
    return true;
  }
  if (s.indexOf('limit') !== -1 && !(/\b(debited|withdrawn|paid|purchase|purchased|spent|credited|deposited)\b/.test(s))) {
    return true;
  }
  return false;
}

/* ---------------- Extract amount, type, subtype, category ---------------- */
function extractAmountAndType(msg) {
  if (!msg) return { amount: null, type: null, subtype: null, category: null, is_otp: false, is_informational: false, is_declined: false };
  // OTP quick check
  if (isOtpMessage(msg)) return { amount: null, type: null, subtype: null, category: null, is_otp: true, is_informational: false, is_declined: false };
  var res = { amount: null, type: null, subtype: null, category: null, is_otp: false, is_informational: false, is_declined: false };
  // amount detection - Rs, INR, ₹
  var mAmt = msg.match(/(?:rs\.?|inr|₹)\s*[:\-]?\s*([0-9,]+(?:\.\d+)?)/i);
  if (mAmt && mAmt[1]) {
    try { res.amount = parseFloat(mAmt[1].replace(/,/g,'')); } catch(e) { res.amount = null; }
  } else {
    // sometimes amounts are like 1,234 or 1,234.00 without currency; optionally capture them if context suggests transaction
    var mAmt2 = msg.match(/\b([0-9]{1,3}(?:,[0-9]{3})+(?:\.\d+)?|\d+(?:\.\d{1,2})?)\b/);
    if (mAmt2 && mAmt2[1] && /\b(debited|withdrawn|purchase|purchased|spent|paid|payment to|withdrawal|atm|bill payment|merchant|credited|deposited)\b/i.test(msg)) {
      try { res.amount = parseFloat(mAmt2[1].replace(/,/g,'')); } catch(e) { res.amount = null; }
    }
  }
  var lower = (msg || '').toLowerCase();
  // declined/failed detection (more conservative)
  if (/\b(txn declined|transaction declined|authorization failed|authorisation failed|auth failed|txn failed|transaction failed)\b/i.test(msg)) {
    res.is_declined = true;
    if (!res.type) res.type = 'declined';
  }
  // credit indicators
  if (/\b(credited|credit\b|deposited|cr[-\s]|neft|rtgs|refund|salary|payroll|payment received|benefit|credited in)\b/i.test(msg)) {
    res.type = 'credit';
  }
  // debit indicators
  if (/\b(debited|withdrawn|purchase|purchased|spent|paid|payment to|withdrawal|atm|bill payment|merchant|merchant payment|card)\b/i.test(msg)) {
    res.type = 'debit';
  }
  // sent disambiguation
  if (/\bsent\b/i.test(lower)) {
    if (/\bsent to\b/i.test(lower) || /\bsent\s+rs\b/i.test(lower) || /\bsent₹|\bsent rs\./i.test(lower)) {
      res.type = 'debit';
    } else if (/\bsent from\b/i.test(lower)) {
      res.type = 'credit';
    }
  }
  if (/\bdr[:\s-]/i.test(msg)) res.type = 'debit';
  if (/\bcr[:\s-]/i.test(msg)) res.type = 'credit';
  // subtype
  if (/\b(card|card ending|cc|credit card|debit card|card no|card \d{4})\b/i.test(msg)) {
    res.subtype = 'card';
    // Don't auto-mark card txns as debit unless explicit - we disallow only explicit card credits later
  } else if (/\b(a\/c|account|ac|a c|a\.c\.)\b/i.test(msg)) {
    res.subtype = 'account';
  } else if (/\b(upi|vpa|gpay|phonepe|paytm|google pay|googlepay)\b/i.test(lower)) {
    res.subtype = 'upi';
  }
  // employer / salary detection
  var employerRegex = /\b(larsen and toubro|l&t|larsen|tata|infosys|wipro|hcl|tech mahindra|payroll|salary|net salary|salary credited)\b/i;
  if (res.type === 'credit' && employerRegex.test(msg)) res.category = 'salary';
  // Deposited to account
  if (/\bdeposited\b/i.test(msg) && /\ba\/c\b|\baccount\b|\bxx\d{2,6}\b/i.test(msg)) res.type = 'credit';
  // Informational detection - only mark informational if no amount present OR explicit informational phrases
  if (isInformationalAlert(msg) && res.amount === null) {
    res.is_informational = true;
  }
  // Fallback: if we have amount and no type, weak heuristics
  if (!res.type && res.amount !== null) {
    if (/\bat\b|\bon\b|\bto\b/i.test(msg) && /\b(at|merchant|shop|store|amazon|flipkart|pay|upi|gpay|phonepe)\b/i.test(msg)) {
      res.type = 'debit';
    } else if (res.is_declined) {
      res.type = 'declined';
    } else {
      // keep null type; processing code can skip if type is unknown
    }
  }
  return res;
}

/* ---------------- Bank extraction & allowed bank regex ---------------- */
function extractBank(sender, msg) {
  var bankMap = {
    'hdfc': 'HDFC Bank',
    'hdfcbk': 'HDFC Bank',
    'hdfc bank': 'HDFC Bank',
    'indb': 'Indian Bank',
    'indbnk': 'Indian Bank',
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
  function mapIt(s) {
    if (!s) return null;
    var t = (''+s).toLowerCase();
    for (var k in bankMap) if (t.indexOf(k) !== -1) return bankMap[k];
    var bankNameMatch = s.match(/([A-Za-z0-9 &]+?)\s+Bank/i);
    if (bankNameMatch) return bankNameMatch[1].trim() + ' Bank';
    return null;
  }
  var fromSender = mapIt(sender);
  if (fromSender) return fromSender;
  var fromMsg = mapIt(msg);
  if (fromMsg) return fromMsg;
  return sender || 'Unknown';
}

function buildAllowedBankRegex() {
  var parts = ALLOWED_BANKS.map(function (b) {
    var e = b.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s*');
    return '(?:\\b' + e + '\\b)';
  });
  return new RegExp(parts.join('|'), 'i');
}

/* ---------- Deduplication helpers ----------- */
function normalizeForHash(s) {
  if (s === null || s === undefined) return '';
  return (''+s).replace(/\s+/g,' ').trim().toLowerCase();
}

function computeMessageHash(sender, message, amount, dt, tz) {
  var s = normalizeForHash(sender);
  var m = normalizeForHash(message);
  // Use amount as fixed 2-dec string if present
  var a = (amount === null || amount === undefined) ? '' : (Number(amount).toFixed(2));
  var t = '';
  if (dt) {
    t = Utilities.formatDate(dt, tz || SpreadsheetApp.getActive().getSpreadsheetTimeZone(), 'yyyy-MM-dd HH:mm');
  }
  return [s, m, a, t].join('||');
}

/* Load only MessageHash column values from History for dedupe set */
function loadExistingHashes() {
  var ss = SpreadsheetApp.getActive();
  var histSheet = ss.getSheetByName('History Transactions');
  var set = {};
  if (!histSheet) return set;
  var lastRow = histSheet.getLastRow();
  if (lastRow < 2) return set;
  var headers = histSheet.getRange(1,1,1,histSheet.getLastColumn()).getValues()[0].map(function(h){ return (''+h).toLowerCase(); });
  var hashIdx = -1;
  for (var i=0;i<headers.length;i++){
    if (headers[i].indexOf('messagehash') !== -1) { hashIdx = i + 1; break; } // 1-based
  }
  if (hashIdx === -1) return set;
  var vals = histSheet.getRange(2, hashIdx, lastRow-1, 1).getValues();
  for (var r=0;r<vals.length;r++){
    var h = vals[r][0];
    if (h) set[''+h] = true;
  }
  return set;
}

/* ---------- Header helpers ---------- */
function findHeaderIndex(headers, possibleNames) {
  // exact normalized match
  var norm = headers.map(function(h){ return (''+h).replace(/\s+/g,' ').trim().toLowerCase(); });
  for (var i=0;i<norm.length;i++){
    for (var j=0;j<possibleNames.length;j++){
      if (norm[i] === possibleNames[j].toLowerCase()) return i;
    }
  }
  return -1;
}

/* ---------- Core: batch processing of RAW (idempotent, no deletes) ---------- */
function processAllRawRowsInBatches() {
  var ss = SpreadsheetApp.getActive();
  var tz = ss.getSpreadsheetTimeZone();
  var props = PropertiesService.getScriptProperties();
  var sheet = ss.getSheetByName(RAW_SHEET_NAME);
  if (!sheet) {
    Logger.log('RAW sheet not found: ' + RAW_SHEET_NAME);
    return;
  }

  var lastRow = sheet.getLastRow();
  var lastCol = sheet.getLastColumn();
  if (lastRow < 2) {
    Logger.log('No data rows in RAW sheet.');
    return;
  }

  // Load existing message hashes for dedupe
  var existingHashes = loadExistingHashes();
  var allowedBankRe = buildAllowedBankRegex();

  // Ensure history/today/sp sheets exist (headers)
  var histSheet = ensureHistorySheet();
  var todaySheet = ensureTodaySheet();
  var spSheet = ensureSuspiciousSheet();

  // Read header row once
  var headers = sheet.getRange(1,1,1,lastCol).getValues()[0];
  var idxTimestamp = findHeaderIndex(headers, ['timestamp','date','time','received']);
  var idxSender = findHeaderIndex(headers, ['sender','from','origin','source']);
  var idxMessage = findHeaderIndex(headers, ['message','sms','text','body']);
  if (idxMessage === -1) idxMessage = lastCol - 1;
  if (idxTimestamp === -1) idxTimestamp = 0;
  if (idxSender === -1) idxSender = 1;

  var startRow = 2;
  var processedTotal = 0;
  var appendedHistory = 0;
  var appendedToday = 0;
  var appendedSuspicious = 0;

  var startTime = new Date().getTime();
  var hardTimeLimit = (5 * 60 - PROCESS_TIME_BUFFER_SECONDS) * 1000; // aim to exit before Apps Script timeout

  while (startRow <= lastRow) {
    // Check runtime
    if ((new Date().getTime() - startTime) > hardTimeLimit) {
      Logger.log('Approaching runtime limit; stopping batch processing. Processed so far: ' + processedTotal);
      break;
    }

    var rowsToRead = Math.min(BATCH_SIZE, lastRow - startRow + 1);
    var data = sheet.getRange(startRow, 1, rowsToRead, lastCol).getValues();

    // Collect rows to append per output sheet in this batch
    var historyRows = [];
    var todayRowsToAppend = [];
    var suspiciousPendingRows = [];

    var today = new Date();
    for (var i=0;i<data.length;i++) {
      var rowNum = startRow + i;
      var row = data[i];

      var rawTimestamp = row[idxTimestamp];
      var sender = row[idxSender];
      var message = row[idxMessage];
      var msgStr = (message === undefined || message === null) ? '' : String(message);

      // Sender allowlist
      if (!sender || !SENDER_REGEX.test(String(sender))) {
        // skip
        continue;
      }

      var dt = parseTimestampStr(rawTimestamp, tz);
      if (!dt) {
        var mm = msgStr.match(/(\d{1,2}[\/-]\d{1,2}[\/-]\d{4})/);
        if (mm) dt = parseTimestampStr(mm[0], tz);
      }

      var at = extractAmountAndType(msgStr);

      // OTP/informational/declined early skipping
      if (at && at.is_otp) continue;
      if (at && at.is_informational && at.amount === null) continue; // keep informative messages if they have an amount
      if (at && at.is_declined) continue;
      if (!at || at.amount === null) continue;

      // Card message but explicitly credit -> skip
      if (at.subtype === 'card' && at.type === 'credit') continue;

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
      // robust allowed bank check: test sender/msg/derived bank
      var keep = allowedBankRe.test((sender||'')) || allowedBankRe.test((msgStr||'')) || allowedBankRe.test((bank||''));
      if (!keep) continue;

      var type = at.type;
      var amount = at.amount;
      var subtype = at.subtype || '';
      var category = at.category || '';
      var suspicious = (amount >= SUSPICIOUS_THRESHOLD) ? 'Yes' : 'No';
      var dtFormatted = dt ? Utilities.formatDate(dt, tz, 'yyyy-MM-dd HH:mm:ss') : '';

      // compute hash
      var rowHash = computeMessageHash(sender, msgStr, amount, dt, tz);
      if (existingHashes[rowHash]) {
        continue;
      }

      // Append to historyRows
      historyRows.push([dtFormatted, bank, type, amount, sender||'', fromMask||'', toMask||'', suspicious, ignoreReason, msgStr, rowNum, rowHash, subtype, category, 'false']);
      existingHashes[rowHash] = true;
      appendedHistory++;

      // Suspicious
      if (suspicious === 'Yes' && !ignoreReason) {
        suspiciousPendingRows.push([Utilities.formatDate(dt, tz, 'yyyy-MM-dd'), dtFormatted, bank, type, amount, sender||'', fromMask||'', toMask||'', msgStr, rowNum, subtype, category]);
        appendedSuspicious++;
      }

      // Today rows: only those with dt = today (spreadsheet tz) and not ignored
      if (dt && Utilities.formatDate(dt, tz, 'yyyy-MM-dd') === Utilities.formatDate(today, tz, 'yyyy-MM-dd') && !ignoreReason) {
        todayRowsToAppend.push([dt, bank, type, amount, sender||'', fromMask||'', toMask||'', suspicious, '', msgStr, rowNum, subtype, category]);
        appendedToday++;
      }

      processedTotal++;
    } // end rows loop

    // Append to History
    if (historyRows.length > 0) {
      histSheet.getRange(histSheet.getLastRow()+1,1,historyRows.length,historyRows[0].length).setValues(historyRows);
      // Format amount column (DateTime=col1, Bank=2, Type=3, Amount=4)
      histSheet.getRange(histSheet.getLastRow()-historyRows.length+1,4,historyRows.length,1).setNumberFormat('#,##0.00');
    }

    if (todayRowsToAppend.length > 0) {
      todaySheet.getRange(todaySheet.getLastRow()+1,1,todayRowsToAppend.length,todayRowsToAppend[0].length).setValues(todayRowsToAppend);
      todaySheet.getRange(todaySheet.getLastRow()-todayRowsToAppend.length+1,1,todayRowsToAppend.length,1).setNumberFormat('yyyy-MM-dd HH:mm:ss');
      todaySheet.getRange(todaySheet.getLastRow()-todayRowsToAppend.length+1,4,todayRowsToAppend.length,1).setNumberFormat('#,##0.00');
    }

    if (suspiciousPendingRows.length > 0) {
      spSheet.getRange(spSheet.getLastRow()+1,1,suspiciousPendingRows.length,suspiciousPendingRows[0].length).setValues(suspiciousPendingRows);
      spSheet.getRange(spSheet.getLastRow()-suspiciousPendingRows.length+1,5,suspiciousPendingRows.length,1).setNumberFormat('#,##0.00');
    }

    // move to next batch
    startRow += rowsToRead;
  } // end while

  // Update Today Summary (best-effort)
  updateTodaySummary();

  Logger.log('Batch processing finished. Processed rows scanned: ' + processedTotal + ' | History appended: ' + appendedHistory + ' | Today appended: ' + appendedToday + ' | Suspicious appended: ' + appendedSuspicious);
}

/* ---------- Ensure output sheets and headers ---------- */
function ensureHistorySheet() {
  var ss = SpreadsheetApp.getActive();
  var histSheet = ss.getSheetByName('History Transactions');
  var outHeader = ['DateTime','Bank','Type','Amount','Sender','FromMask','ToMask','Suspicious','IgnoreReason','Message','SourceRow','MessageHash','Subtype','Category','is_deleted'];
  if (!histSheet) {
    histSheet = ss.insertSheet('History Transactions');
    histSheet.getRange(1,1,1,outHeader.length).setValues([outHeader]);
    return histSheet;
  }
  // Ensure all headers exist; if missing append them at end
  var currentHeaders = histSheet.getRange(1,1,1,histSheet.getLastColumn()).getValues()[0];
  var lower = currentHeaders.map(function(h){return (''+h).toLowerCase();});
  for (var i=0;i<outHeader.length;i++){
    if (lower.indexOf(outHeader[i].toLowerCase()) === -1) {
      histSheet.getRange(1, histSheet.getLastColumn()+1).setValue(outHeader[i]);
    }
  }
  return histSheet;
}

function ensureTodaySheet() {
  var ss = SpreadsheetApp.getActive();
  var todaySheet = ss.getSheetByName('Today Transactions');
  var outHeaderT = ['DateTime','Bank','Type','Amount','Sender','FromMask','ToMask','Suspicious','IgnoreReason','Message','SourceRow','Subtype','Category'];
  if (!todaySheet) {
    todaySheet = ss.insertSheet('Today Transactions');
    todaySheet.getRange(1,1,1,outHeaderT.length).setValues([outHeaderT]);
    return todaySheet;
  }
  var ch = todaySheet.getRange(1,1,1,todaySheet.getLastColumn()).getValues()[0].map(function(h){return (''+h).toLowerCase();});
  for (var i=0;i<outHeaderT.length;i++){
    if (ch.indexOf(outHeaderT[i].toLowerCase()) === -1) {
      todaySheet.getRange(1, todaySheet.getLastColumn()+1).setValue(outHeaderT[i]);
    }
  }
  return todaySheet;
}

function ensureSuspiciousSheet() {
  var ss = SpreadsheetApp.getActive();
  var sp = ss.getSheetByName('Suspicious Pending');
  var spHeader = ['Date','DateTime','Bank','Type','Amount','Sender','FromMask','ToMask','Message','SourceRow','Subtype','Category'];
  if (!sp) {
    sp = ss.insertSheet('Suspicious Pending');
    sp.getRange(1,1,1,spHeader.length).setValues([spHeader]);
    return sp;
  }
  var ch = sp.getRange(1,1,1,sp.getLastColumn()).getValues()[0].map(function(h){return (''+h).toLowerCase();});
  for (var i=0;i<spHeader.length;i++){
    if (ch.indexOf(spHeader[i].toLowerCase()) === -1) {
      sp.getRange(1, sp.getLastColumn()+1).setValue(spHeader[i]);
    }
  }
  return sp;
}

/* ---------- Today Summary ---------- */
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
    ['Allowed Banks (keywords)', buildAllowedBankRegex().toString()]
  ];
  sumSheet.getRange(1,1,rows.length,2).setValues(rows);
  if (rows.length >= 2) sumSheet.getRange(2,2,2,1).setNumberFormat('#,##0.00');
}

/* ---------- Nightly final sweep + purge RAW ---------- */
function runNightlyCleanup() {
  var ss = SpreadsheetApp.getActive();
  var tz = ss.getSpreadsheetTimeZone();
  var today = new Date();
  var todayStr = Utilities.formatDate(today, tz, 'yyyy-MM-dd');

  // Final sweep to capture any late rows
  processAllRawRowsInBatches();

  // Purge RAW: clear all rows except header (safer than deleting rows)
  var sheet = ss.getSheetByName(RAW_SHEET_NAME);
  if (sheet) {
    var lastRow = sheet.getLastRow();
    var lastCol = sheet.getLastColumn();
    if (lastRow >= 2) {
      try {
        sheet.getRange(2,1,lastRow-1,lastCol).clearContent();
        Logger.log('Cleared RAW sheet contents (kept header).');
      } catch (e) {
        Logger.log('Error clearing RAW sheet: ' + e);
      }
    }
  }

  // Clear Today Transactions and Today Summary (prepare for next day)
  var todaySheet = ss.getSheetByName('Today Transactions');
  if (todaySheet) {
    var outHeaderT = ['DateTime','Bank','Type','Amount','Sender','FromMask','ToMask','Suspicious','IgnoreReason','Message','SourceRow','Subtype','Category'];
    todaySheet.clearContents();
    todaySheet.getRange(1,1,1,outHeaderT.length).setValues([outHeaderT]);
  }
  var sumSheet = ss.getSheetByName('Today Summary');
  if (sumSheet) sumSheet.clearContents();

  // Remove today's entries from Suspicious Pending
  var sp = ss.getSheetByName('Suspicious Pending');
  if (sp) {
    var spVals = sp.getDataRange().getValues();
    if (spVals.length > 1) {
      var rowsToKeep = [spVals[0]]; // header
      for (var r=1; r<spVals.length; r++){
        if ((spVals[r][0] || '') !== todayStr) rowsToKeep.push(spVals[r]);
      }
      sp.clearContents();
      if (rowsToKeep.length > 0) sp.getRange(1,1,rowsToKeep.length,rowsToKeep[0].length).setValues(rowsToKeep);
    }
  }

  Logger.log('Nightly cleanup finished for ' + todayStr);
}

/* ---------- Triggers / installers ---------- */
function installTriggers() {
  deleteTriggersForHandler('processAllRawRowsInBatches');
  deleteTriggersForHandler('runNightlyCleanup');
  // hourly sweep
  ScriptApp.newTrigger('processAllRawRowsInBatches').timeBased().everyHours(1).create();
  // nightly cleanup - runs at 23:00 project timezone; set project timezone to Asia/Kolkata for 23:00 IST
  ScriptApp.newTrigger('runNightlyCleanup').timeBased().atHour(23).everyDays(1).create();
  SpreadsheetApp.getUi().alert('Triggers installed: hourly processing and nightly cleanup at 23:00 (project timezone). Set project timezone to Asia/Kolkata for 23:00 IST.');
}

function uninstallTriggers() {
  deleteTriggersForHandler('processAllRawRowsInBatches');
  deleteTriggersForHandler('runNightlyCleanup');
  SpreadsheetApp.getUi().alert('Triggers removed.');
}

function deleteTriggersForHandler(fn) {
  var all = ScriptApp.getProjectTriggers();
  for (var i=0;i<all.length;i++){
    if (all[i].getHandlerFunction() === fn) ScriptApp.deleteTrigger(all[i]);
  }
}

/* ---------- Optional manual helpers ---------- */
function processAllNowFromMenu() {
  // helper that can be bound to a custom menu to run immediate processing
  processAllRawRowsInBatches();
  SpreadsheetApp.getUi().alert('Manual processing completed. Check logs for details.');
}

/* ---------- Legacy onEdit/onChange handlers are intentionally no-ops (we use scheduled processing)
   If you want to keep instantaneous processing for manual edits, you can call processAllRawRowsInBatches()
   from these handlers, but be cautious about running heavy processing on each edit. ---------- */
function onEditHandler(e) {
  // no-op by default to avoid processing on each edit
}

function onChangeHandler(e) {
  // no-op by default to avoid processing on each change
}

/* ---------- Init helper to ensure pointer / sheets exist ---------- */
function initSheetsAndPointer() {
  ensureHistorySheet();
  ensureTodaySheet();
  ensureSuspiciousSheet();
  var ss = SpreadsheetApp.getActive();
  var sheet = ss.getSheetByName(RAW_SHEET_NAME);
  if (!sheet) SpreadsheetApp.getUi().alert('Raw sheet "'+RAW_SHEET_NAME+'" not found. Create it and run init again.');
  else SpreadsheetApp.getUi().alert('Sheets ensured. Ready.');
}
