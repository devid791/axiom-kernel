/*
 * axiom_aliced_json.h — minimal, dependency-free JSON value + parser +
 * serializer for aliced (tools/axiom_aliced.cpp). C++17, no exceptions
 * thrown by this code (parse errors are returned), no external libraries.
 *
 * Design constraints (see docs/deepseek_v4_flash_cluster/aliced_design.md):
 *   - object member INSERTION ORDER is preserved (required for byte-stable
 *     re-serialization of the tools prompt embedded in the DS4 chat template,
 *     matching Python dict/json.dumps behavior in dist/scripts/axiomd);
 *   - nesting depth is limited (ALICED_JSON_MAX_DEPTH) — no recursion bombs;
 *   - strings are stored/emitted as raw UTF-8 bytes (json.dumps
 *     ensure_ascii=False equivalence); \uXXXX escapes (incl. surrogate
 *     pairs) are decoded to UTF-8 on parse;
 *   - integers and doubles are kept distinct (Python int vs float), and
 *     doubles serialize via shortest round-trip, "N.0" for integral values
 *     (Python repr style);
 *   - the value type is a tagged struct with parallel key/value vectors for
 *     objects: std::vector<ajson> with the incomplete element type is
 *     guaranteed by C++17, std::pair<std::string, ajson> is not.
 */
#ifndef AXIOM_ALICED_JSON_H
#define AXIOM_ALICED_JSON_H

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifndef ALICED_JSON_MAX_DEPTH
#define ALICED_JSON_MAX_DEPTH 64
#endif

struct ajson {
    enum kind_t { NUL = 0, BOOL, INT, NUM, STR, ARR, OBJ };

    kind_t kind = NUL;
    bool b = false;
    long long i = 0;
    double d = 0.0;
    std::string s;
    std::vector<ajson> arr;        /* ARR elements */
    std::vector<std::string> keys; /* OBJ member names, insertion order */
    std::vector<ajson> vals;       /* OBJ member values, parallel to keys */

    /* ---- constructors ---- */
    static ajson jnull() { return ajson(); }
    static ajson jbool(bool v) { ajson j; j.kind = BOOL; j.b = v; return j; }
    static ajson jint(long long v) { ajson j; j.kind = INT; j.i = v; return j; }
    static ajson jnum(double v) { ajson j; j.kind = NUM; j.d = v; return j; }
    static ajson jstr(const char *v) { ajson j; j.kind = STR; j.s = v ? v : ""; return j; }
    static ajson jstr(std::string v) { ajson j; j.kind = STR; j.s = std::move(v); return j; }
    static ajson jarr() { ajson j; j.kind = ARR; return j; }
    static ajson jobj() { ajson j; j.kind = OBJ; return j; }

    /* ---- predicates ---- */
    bool is_null() const { return kind == NUL; }
    bool is_bool() const { return kind == BOOL; }
    bool is_int() const { return kind == INT; }
    bool is_number() const { return kind == INT || kind == NUM; }
    bool is_string() const { return kind == STR; }
    bool is_array() const { return kind == ARR; }
    bool is_object() const { return kind == OBJ; }

    double number() const { return kind == INT ? (double)i : (kind == NUM ? d : 0.0); }

    /* ---- object access ---- */
    const ajson *get(const char *key) const {
        if (kind != OBJ || !key) return nullptr;
        for (size_t n = 0; n < keys.size(); n++) {
            if (keys[n] == key) return &vals[n];
        }
        return nullptr;
    }
    bool has(const char *key) const { return get(key) != nullptr; }

    /* replace-or-append, preserving original position on replace
     * (mirrors Python dict assignment) */
    void set(const std::string &key, ajson v) {
        if (kind != OBJ) { kind = OBJ; keys.clear(); vals.clear(); arr.clear(); s.clear(); }
        for (size_t n = 0; n < keys.size(); n++) {
            if (keys[n] == key) { vals[n] = std::move(v); return; }
        }
        keys.push_back(key);
        vals.push_back(std::move(v));
    }

    void push(ajson v) {
        if (kind != ARR) { kind = ARR; arr.clear(); keys.clear(); vals.clear(); s.clear(); }
        arr.push_back(std::move(v));
    }
};

/* ---------------------------------------------------------------- parse -- */

struct ajson_parser {
    const char *p;
    const char *end;
    std::string err;

    ajson_parser(const char *text, size_t len) : p(text), end(text + len) {}

    bool fail(const char *msg) {
        if (err.empty()) err = msg;
        return false;
    }

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }

    static void utf8_append(std::string &out, unsigned long cp) {
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }

    bool hex4(unsigned &out) {
        if (end - p < 4) return fail("truncated \\u escape");
        unsigned v = 0;
        for (int n = 0; n < 4; n++) {
            char c = p[n];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
            else return fail("bad \\u escape");
        }
        p += 4;
        out = v;
        return true;
    }

    bool parse_string(std::string &out) {
        if (p >= end || *p != '"') return fail("expected string");
        p++;
        out.clear();
        while (p < end) {
            unsigned char c = (unsigned char)*p;
            if (c == '"') { p++; return true; }
            if (c == '\\') {
                p++;
                if (p >= end) return fail("truncated escape");
                char e = *p++;
                switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF && end - p >= 6 &&
                        p[0] == '\\' && p[1] == 'u') {
                        const char *save = p;
                        p += 2;
                        unsigned lo = 0;
                        if (!hex4(lo)) return false;
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            unsigned long full =
                                0x10000ul + (((unsigned long)cp - 0xD800ul) << 10) +
                                ((unsigned long)lo - 0xDC00ul);
                            utf8_append(out, full);
                            break;
                        }
                        /* not a low surrogate: emit first, rewind, continue */
                        p = save;
                    }
                    /* lone surrogates are encoded as-is (WTF-8), documented */
                    utf8_append(out, cp);
                    break;
                }
                default:
                    return fail("bad escape character");
                }
                continue;
            }
            if (c < 0x20) return fail("raw control character in string");
            out += (char)c;  /* UTF-8 passthrough */
            p++;
        }
        return fail("unterminated string");
    }

    bool parse_number(ajson &out) {
        const char *start = p;
        if (p < end && *p == '-') p++;
        if (p >= end || *p < '0' || *p > '9') return fail("bad number");
        if (*p == '0') {
            p++;
        } else {
            while (p < end && *p >= '0' && *p <= '9') p++;
        }
        bool is_int = true;
        if (p < end && *p == '.') {
            is_int = false;
            p++;
            if (p >= end || *p < '0' || *p > '9') return fail("bad number fraction");
            while (p < end && *p >= '0' && *p <= '9') p++;
        }
        if (p < end && (*p == 'e' || *p == 'E')) {
            is_int = false;
            p++;
            if (p < end && (*p == '+' || *p == '-')) p++;
            if (p >= end || *p < '0' || *p > '9') return fail("bad number exponent");
            while (p < end && *p >= '0' && *p <= '9') p++;
        }
        std::string lex(start, (size_t)(p - start));
        if (is_int) {
            errno = 0;
            char *lend = nullptr;
            long long v = strtoll(lex.c_str(), &lend, 10);
            if (errno == 0 && lend && *lend == '\0') {
                out = ajson::jint(v);
                return true;
            }
            /* out of int64 range: fall through to double */
        }
        errno = 0;
        double dv = strtod(lex.c_str(), nullptr);
        if (!std::isfinite(dv)) return fail("number out of range");
        out = ajson::jnum(dv);
        return true;
    }

    bool parse_value(ajson &out, int depth) {
        if (depth > ALICED_JSON_MAX_DEPTH) return fail("nesting too deep");
        skip_ws();
        if (p >= end) return fail("unexpected end of input");
        char c = *p;
        if (c == '{') {
            p++;
            out = ajson::jobj();
            skip_ws();
            if (p < end && *p == '}') { p++; return true; }
            while (true) {
                skip_ws();
                std::string key;
                if (!parse_string(key)) return false;
                skip_ws();
                if (p >= end || *p != ':') return fail("expected ':'");
                p++;
                ajson v;
                if (!parse_value(v, depth + 1)) return false;
                out.set(key, std::move(v));
                skip_ws();
                if (p < end && *p == ',') { p++; continue; }
                if (p < end && *p == '}') { p++; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            p++;
            out = ajson::jarr();
            skip_ws();
            if (p < end && *p == ']') { p++; return true; }
            while (true) {
                ajson v;
                if (!parse_value(v, depth + 1)) return false;
                out.push(std::move(v));
                skip_ws();
                if (p < end && *p == ',') { p++; continue; }
                if (p < end && *p == ']') { p++; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            std::string sv;
            if (!parse_string(sv)) return false;
            out = ajson::jstr(std::move(sv));
            return true;
        }
        if (c == 't') {
            if (end - p >= 4 && memcmp(p, "true", 4) == 0) { p += 4; out = ajson::jbool(true); return true; }
            return fail("bad literal");
        }
        if (c == 'f') {
            if (end - p >= 5 && memcmp(p, "false", 5) == 0) { p += 5; out = ajson::jbool(false); return true; }
            return fail("bad literal");
        }
        if (c == 'n') {
            if (end - p >= 4 && memcmp(p, "null", 4) == 0) { p += 4; out = ajson::jnull(); return true; }
            return fail("bad literal");
        }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(out);
        return fail("unexpected character");
    }
};

/* Parse an entire buffer as one JSON document (trailing whitespace only). */
inline bool ajson_parse(const std::string &text, ajson &out, std::string &err) {
    ajson_parser ps(text.data(), text.size());
    if (!ps.parse_value(out, 0)) {
        err = ps.err.empty() ? "invalid JSON" : ps.err;
        return false;
    }
    ps.skip_ws();
    if (ps.p != ps.end) {
        err = "trailing data after JSON value";
        return false;
    }
    return true;
}

/* ----------------------------------------------------------------- dump -- */

inline void ajson_dump_string_body(const std::string &s, std::string &out) {
    static const char *hexdig = "0123456789abcdef";
    for (size_t n = 0; n < s.size(); n++) {
        unsigned char c = (unsigned char)s[n];
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                out += "\\u00";
                out += hexdig[(c >> 4) & 0xF];
                out += hexdig[c & 0xF];
            } else {
                out += (char)c;  /* raw UTF-8 passthrough (ensure_ascii=False) */
            }
        }
    }
}

inline void ajson_dump_string(const std::string &s, std::string &out) {
    out += '"';
    ajson_dump_string_body(s, out);
    out += '"';
}

/* Shortest round-trip double, Python-repr style ("1.0", "0.25", "1e+20").
 * Non-finite values serialize as null (we never produce them ourselves). */
inline void ajson_dump_double(double v, std::string &out) {
    if (!std::isfinite(v)) { out += "null"; return; }
    char buf[40];
    buf[0] = '\0';
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(buf, sizeof(buf), "%.*g", prec, v);
        if (strtod(buf, nullptr) == v) break;
    }
    if (!strpbrk(buf, ".eEnN")) {
        size_t len = strlen(buf);
        if (len + 3 <= sizeof(buf)) { buf[len] = '.'; buf[len + 1] = '0'; buf[len + 2] = '\0'; }
    }
    out += buf;
}

inline void ajson_dump(const ajson &v, std::string &out) {
    switch (v.kind) {
    case ajson::NUL: out += "null"; break;
    case ajson::BOOL: out += v.b ? "true" : "false"; break;
    case ajson::INT: {
        char buf[32];
        snprintf(buf, sizeof(buf), "%lld", v.i);
        out += buf;
        break;
    }
    case ajson::NUM: ajson_dump_double(v.d, out); break;
    case ajson::STR: ajson_dump_string(v.s, out); break;
    case ajson::ARR:
        out += '[';
        for (size_t n = 0; n < v.arr.size(); n++) {
            if (n) out += ',';
            ajson_dump(v.arr[n], out);
        }
        out += ']';
        break;
    case ajson::OBJ:
        out += '{';
        for (size_t n = 0; n < v.keys.size(); n++) {
            if (n) out += ',';
            ajson_dump_string(v.keys[n], out);
            out += ':';
            ajson_dump(v.vals[n], out);
        }
        out += '}';
        break;
    }
}

inline std::string ajson_dumps(const ajson &v) {
    std::string out;
    out.reserve(256);
    ajson_dump(v, out);
    return out;
}

#endif /* AXIOM_ALICED_JSON_H */
