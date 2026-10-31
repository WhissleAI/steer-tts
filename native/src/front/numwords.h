// Ports of steertts_edge/edge_front/numwords.py (num2words / inflect subsets) and normnum.py (g2p_en
// normalize_numbers). Integers are handled as decimal digit strings (Python ints are unbounded); Python exceptions
// are mirrored as C++ exceptions carrying Python's message:
//   ValueError -> std::invalid_argument, OverflowError -> std::overflow_error, IndexError -> std::out_of_range.
#pragma once
#include <string>

#include "pyre.h"
#include "ucd.h"

namespace steertts {
namespace detail {

// Python int(s) for a string of (Unicode) decimal digits: ascii digits without leading zeros ("0" for zero).
// Raises the CPython 3.11 4300-digit limit ValueError like int() does.
std::string py_int_digits(const u32str& s, const Ucd& u);

class Numwords {
 public:
  explicit Numwords(const Ucd* u);
  // num2words(n) / num2words(n, to="ordinal"); digits = normalised non-negative decimal string
  std::string cardinal(const std::string& digits) const;
  std::string ordinal(const std::string& digits) const;
  std::string cardinal_float(double v) const;  // num2words(<float>)
  // inflect.engine().number_to_words(s, andword, zero, group) as numwords.inflect_words (s: str(int) or "21st")
  std::string inflect_words(const std::string& s, const std::string& andword = "and", const std::string& zero = "zero",
                            int group = 0) const;
  u32str normalize_numbers(const u32str& text) const;  // normnum.normalize_numbers

 private:
  std::string i_enword(std::string num, int group, const std::string& zero, const std::string& one,
                       const std::string& andword) const;
  std::string inflect_ordinal_words(const std::string& w) const;
  std::string expand_number(const u32str& m) const;
  std::string expand_dollars(const u32str& m) const;
  const Ucd* u_;
  Regex ord_suffix_, pair_, last_digit_, cl1_, cl2_, cl3_, ws_, final_one_, last_word_;
  Regex comma_number_, pounds_, dollars_, decimal_, ordinal_re_, number_;
};

}  // namespace detail
}  // namespace steertts
