#include <iostream>
#include <string>
#include <variant>
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <algorithm> // generate?, transform, unique, reverse, rotate
#include <numeric>   // iota, accumulate, inner_product, partial_sum
#include <cmath>     // pow

#include <stdio.h>
#include <stdlib.h>
// #include <ncurses.h>

#include "termcolor.hpp"
#include "expected.hpp"

using namespace std::string_literals;

// COLOR SCHEME

#define COLOR_HIGHLIGHT  termcolor::red
#define COLOR_ERROR      termcolor::red
#define COLOR_INPUT      termcolor::green
#define COLOR_OUTPUT     termcolor::yellow


// forward declaration
class noun;

// Subject Type
using scalar = int;
using vector = std::vector<int>; // TODO support chars
using matrix = std::vector<vector>;
using nested_vector = std::vector<noun>;

using ad_verb = std::string;

enum class ParenType : bool {
    LEFT,
    RIGHT
};

struct pronoun     { std::string name;  };
struct adverb      { std::string glyph; };
struct verb        { std::string glyph; };
struct punctuation { ParenType type;    };
struct copula      {}; // no need to store ←

using token = 
    std::variant<
        noun,
        verb,
        adverb,
        copula,
        pronoun,
        punctuation>;

using error = tl::unexpected<std::string>;

using expected_noun = tl::expected<noun, std::string>;

enum class noun_type {
    NESTED_VECTOR,
    SCALAR,
    VECTOR,
    MATRIX
};

using noun_data =
    std::variant<
        scalar,
        vector,
        matrix,
        nested_vector>;

template <typename T>
auto get_shapes(T const& t) {
    std::vector<std::size_t> shapes;
    if constexpr (std::is_integral_v<decltype(t)>) return shapes;
    shapes.push_back(t.size());
    if constexpr (std::is_integral_v<decltype(t.at(0))>) return shapes;
    shapes.push_back(t[0].size());
    // if constexpr (std::is_integral_v<decltype(t.at(0).at(0))>) return shapes;
    // shapes.push_back(t[0][0].size());
    return shapes;
}

class noun {
    noun_type                _type; // try to add const
    std::vector<std::size_t> _shape;
    std::size_t              _rank;

    noun_data _data;

public:

    noun(scalar val) :
        _data(val),
        _type(noun_type::SCALAR),
        _shape({}),
        _rank(0)
    {
        assert(_shape.size() == _rank && "Shape size doesn't equal.");
    }

    noun(vector arr) :
        _data(arr),
        _type(noun_type::VECTOR),
        _shape({arr.size()}),
        _rank(1)
    {
        assert(_shape.size() == _rank && "Shape size doesn't equal.");
    }

    noun(matrix m) :
        _data(m),
        _type(noun_type::MATRIX),
        _shape(get_shapes(m)),
        _rank(2)
    {
        assert(_shape.size() == _rank && "Shape size doesn't equal.");
    }

    noun(noun_type, nested_vector nv) :
        _data(nv),
        _type(noun_type::NESTED_VECTOR),
        _shape({nv.size()}), //get_shapes_nv(nv)
        _rank(1)
    {
        assert(_shape.size() == _rank && "Shape size doesn't equal.");
    }

    [[ nodiscard ]] constexpr auto type() const noexcept { return _type; };
    [[ nodiscard ]] constexpr auto rank() const noexcept { return _rank; };

    [[ nodiscard ]] auto shape() const noexcept -> std::vector<std::size_t> const& { return _shape; };

    [[ nodiscard ]] auto data() -> noun_data& { return _data; };
    [[ nodiscard ]] auto data() const noexcept { return _data; };
};

std::unordered_map<std::string, noun> pronoun_list;

auto to_string(noun_type const& n) -> std::string {
    switch (n) {
    case noun_type::NESTED_VECTOR: return "Nested vector";
    case noun_type::SCALAR:        return "Scalar";
    case noun_type::VECTOR:        return "Vector";
    case noun_type::MATRIX:        return "Matrix";
    default:                       return "FAIL";
    }
}

template <typename T,
          typename std::enable_if_t<std::is_integral_v<T>>* = nullptr>
auto to_string(T t) -> std::string {
    return std::to_string(t);
}

auto left_pad(int len, std::string const& str) {
    return std::string(len - str.size(), ' ') + str;
}

template <typename T>
auto to_string_with_pads(std::vector<T>   const& values,
                         std::vector<int> const& widths) -> std::string {
    return std::inner_product(
        std::cbegin(values),
        std::cend(values),
        std::cbegin(widths),
        std::string{},
        [] (auto const& acc, auto const& str) {
            return acc + str; },
        [] (auto const& value, auto const& width) {
            return left_pad(width + 1, std::to_string(value)); });
}

template <typename T>
auto to_string(std::vector<T> const& v) -> std::string {
    if (v.empty())
        return "Empty";
    if constexpr (std::is_integral_v<T>) {
        auto res = std::accumulate(
            std::next(std::cbegin(v)),
            std::cend(v),
            std::string{to_string(v.front())},
            [] (auto const& acc, auto e) {
                return acc + " " + to_string(e); });
        return res;
    } else if constexpr (!std::is_integral_v<T>
                      && !std::is_same_v<T, noun>) {

        // could use range-v3 transpose here
        auto max_digits_per_column = std::vector<int>(v.front().size(), 0);
        int const rows = v.size();
        int const cols = v.front().size();
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < rows; ++j) {
                max_digits_per_column[i] = std::max(
                    max_digits_per_column[i], (int)std::to_string(v[j][i]).size());
            }
        }

        auto res = std::accumulate(
            std::next(std::cbegin(v)),
            std::cend(v),
            std::string{to_string_with_pads(v.front(), max_digits_per_column)},
            [first = true, &max_digits_per_column]
            (auto const& acc, auto e) mutable {
                first = false;
                auto const spaces = first ? ""s : "    "s;
                return acc + "\n\r" + spaces +
                    to_string_with_pads(e, max_digits_per_column);
            });

        return res;
    } else {
        return "not implemented yet";
    }
}

auto to_string(noun_data const& n) -> std::string {
    if (std::holds_alternative<scalar>(n)) {
        auto const val = std::get<scalar>(n);
        return std::to_string(val);
    } else if (std::holds_alternative<vector>(n)) {
        auto const& v = std::get<vector>(n);
        return to_string(v);
    } else if (std::holds_alternative<matrix>(n)) {
        auto const& m = std::get<matrix>(n);
        return to_string(m);
    } else {
        auto const& nv = std::get<nested_vector>(n);

        std::vector<std::size_t> lens;
        std::transform(
            std::cbegin(nv),
            std::cend(nv),
            std::back_inserter(lens),
            [] (auto const& e) {
                auto const& vec = std::get<vector>(e.data());
                return vec.size(); });

        auto top_bottom = std::accumulate(
            std::cbegin(lens),
            std::cend(lens),
            std::string{"+"},
            [] (auto& s, auto e) {
                s.resize(s.size() + 2*e + 1, '-');
                s += "+";
                return s; });

        auto middle = std::accumulate(
            std::cbegin(nv),
            std::cend(nv),
            std::string{"| "},
            [] (auto acc, auto e) {
                return acc + to_string(e.data()) + " | ";
            });

        return top_bottom + "\n\r    " +
               middle     + "\n\r    " +
               top_bottom;
    }
}

auto to_string(noun const& n) -> std::string {
    return "Type:  "s + to_string(n.type()) + "\n\r"s
           "Shape: "s + to_string(n.shape()) + "\n\r"s
           "Rank:  "s + std::to_string(n.rank()) + "\n\r"s
           "Data:  "s + "\n\r"s + to_string(n.data());
}

std::ostream& operator<<(std::ostream& os, noun const& n) {
    // os.flush();
    os << COLOR_OUTPUT << to_string(n.data());
    return os;
}

// end of nount /

enum ASCII {
    BACKSPACE = 8,
    RETURN    = 13,
    UP_ARROW  = 24
};

std::unordered_set<char> ascii_apl = { '/', '\\', '+', '-', '*', ',',
                                       '=', '(', ')', '|', '~' , '<',
                                       '>' };

auto is_adverb(std::string s) -> bool {
    return s == "/" || s == "\\";
}

auto is_adverb(char c) -> bool {
    return c == '/' || c == '\\';
}

namespace APLCharSet {
    auto const IOTA                  = "⍳";
    auto const INDEX_OF              = IOTA; // dyadic
    auto const COMPOSE               = "∘";
    auto const OUTER_PRODUCT         = "∘.";
    auto const RANK                  = "⍤";
    auto const DROP                  = "↓";
    auto const SPLIT                 = DROP;
    auto const TAKE                  = "↑";
    auto const MIX                   = TAKE;
    auto const REDUCE                = "/";
    auto const REPLICATE             = REDUCE;
    auto const SCAN                  = "\\";
    auto const EXPAND                = SCAN;
    auto const REVERSE               = "⌽";
    auto const ROTATE                = REVERSE;
    auto const UNIQUE                = "∪";
    auto const UNION                 = UNIQUE;
    auto const RAVEL                 = ",";
    auto const CATENATE              = RAVEL;
    auto const CEILING               = "⌈";
    auto const MAXIMUM               = CEILING;
    auto const FLOOR                 = "⌊";
    auto const MINIMUM               = FLOOR;
    auto const ENCLOSE               = "⊂";
    auto const PARTITIONED_ENCLOSE   = ENCLOSE;
    auto const FIRST                 = "⊃"; // DISCLOSE ??
    auto const PICK                  = FIRST;
    auto const NEST                  = "⊆";
    auto const PARTITION             = NEST;
    auto const NOT_EQUAL_TO          = "≠";
    auto const EQUAL_TO              = "=";
    auto const EACH                  = "¨";
    auto const DEPTH                 = "≡";
    auto const MATCH                 = DEPTH;
    auto const ABSOLUTE_VALUE        = "|";
    auto const RESIDUE               = ABSOLUTE_VALUE;
    auto const SHAPE                 = "⍴";
    auto const RESHAPE               = SHAPE;
    auto const NEGATE                = "-";
    auto const SUBTRACT              = NEGATE;
    auto const CONJUGATE             = "+";
    auto const ADD                   = CONJUGATE;
    auto const SIGN_OF               = "×";
    auto const MULTIPLY              = SIGN_OF;
    auto const RECIPROCAL            = "÷";
    auto const DIVIDE                = RECIPROCAL;
    auto const LOGICAL_AND           = "∧";
    auto const LOGICAL_OR            = "∨";
    auto const ENLIST                = "∊";
    auto const MEMBERSHIP            = ENLIST;
    auto const FIND                  = "⍷";
    auto const LEFT_ARROW            = "←";
    auto const NOT                   = "~";
    auto const WITHOUT               = NOT;
    auto const EXPONENTIAL           = "*";
    auto const POWER                 = EXPONENTIAL;
    auto const LESS_THAN             = "<";
    auto const LESS_THAN_OR_EQUAL    = "≤";
    auto const GREATER_THAN          = ">";
    auto const GREATER_THAN_OR_EQUAL = "≥";
}

auto getAplCharFromShortCut(char c) -> std::string {
    using namespace APLCharSet;
    switch (c) {
        case 'i': return IOTA;
        case 'j': return COMPOSE;
        case 'J': return RANK;
        case 'y': return TAKE;
        case 'u': return DROP;
        case '%': return REVERSE;
        case 'v': return UNIQUE;
        case 's': return CEILING;
        case 'x': return FIRST;
        case 'z': return ENCLOSE;
        case 'Z': return NEST;
        case '8': return NOT_EQUAL_TO;
        case '1': return EACH;
        case ':': return DEPTH;
        case 'r': return SHAPE;
        case '-': return SIGN_OF;
        case '=': return RECIPROCAL;
        case '0': return LOGICAL_AND;
        case '9': return LOGICAL_OR;
        case 'e': return ENLIST;
        case 'E': return FIND;
        case 'd': return FLOOR;
        case '[': return LEFT_ARROW;
        case 'p': return POWER;
        case '4': return LESS_THAN_OR_EQUAL;
        case '6': return GREATER_THAN_OR_EQUAL;
        default:  return "unkown character"s + c;
    }
}

auto getAplCharFromString(std::string s) -> std::string {
    using namespace APLCharSet;
    if (s == "iota"s) return IOTA;
        // case 'j': return "∘"s;
        // case 'J': return "⍤"s;
    else if (s == "drop"s)    return DROP;
    else if (s == "take"s)    return TAKE;
    else if (s == "reverse"s) return REVERSE;
    else if (s == "rotate"s)  return ROTATE;
    else return "unkown string"s + s;
}

// replace with when range v3 is fixed
// auto tokenize(std::string s) {
//     s = " " + s + " ";
//     return s
//         | rv::sliding(3)
//         | rv::filter([](auto rng) {
//             if (rng[1] != ' ') return true;
//             return std::isdigit(rng[0]) && std::isdigit(rng[2]); })
//         | rv::transform([](auto rng) { return rng[1]; })
//         | rv::group_by([](auto a, auto b) {
//             return !std::isalpha(a) && !std::isalpha(b); });
// }

// void print_flat_tokens(std::stack<token>); // FOR DEBUGGING

auto get_number(std::string_view s, int i) -> std::pair<int, int> {
    int num = s[i] - '0';
    while (i != s.size() - 1 && std::isdigit(s[i+1]))
        num = num * 10 + (s[++i] - '0');
    return { num, i };
}

auto tokenize(std::string_view s) -> std::stack<token> {
    // should trim first, guarantee 1st and last won't be spaces
    std::stack<token> stack;
    std::vector<int> num_literals;
    for (int i = 0; i < s.size(); ++i) {
        // print_flat_tokens(stack); // FOR DEBUGGING
        auto c = s[i];
        if (std::isdigit(c)) {
            auto [num, j] = get_number(s, i);
            if (j == s.size() - 1 || s[j+1] != ' ')
                stack.push(token{noun{num}}); // scalar
            else {
                num_literals.push_back(num);
                while (j != s.size() - 1 && s[j+1] == ' ') {
                    j += 2;
                    std::tie(num, j) = get_number(s, j);
                    num_literals.push_back(num);
                }
                stack.push(token{noun{num_literals}}); // vector
                num_literals.clear();
            }
            i = j;
        } else if (std::isalpha(c)) {
            auto var = std::string{c};
            while (i + 1 < s.size() && std::isalpha(s[i+1]))
                var += s[i++];
            stack.push(token{pronoun{var}});
        } else if (c != ' ') {
            if (c == '(') {
                stack.push(token{punctuation{ParenType::LEFT}});
            } else if (c == ')') {
                stack.push(token{punctuation{ParenType::RIGHT}});
            } else if (is_adverb(c) &&
                std::holds_alternative<verb>(stack.top())) {
                stack.push(token{adverb{std::string{c}}});
            } else if (ascii_apl.count(c)) {
                stack.push(token{verb{std::string{c}}});
            } else {
                // deal with APL char 3 char width
                auto const delta = s.substr(i, 2) == "×" ? 2 : 3;
                if (s.substr(i, delta) == APLCharSet::LEFT_ARROW) {
                    stack.push(token{copula{}});
                } else {
                    stack.push(token{verb{std::string{s.substr(i, delta)}}});
                }
                i += delta - 1;
            }
        } else {
            assert(c == ' ' && "Expected a space");
        }
    }

    // TODO or parentheses
    // assert(std::holds_alternative<noun>(stack.top())
    //     && "Top element of token stack should always be a noun");

    return stack;
}

void print_flat_tokens(std::stack<token> tokens) {
    std::vector<token> v;

    // std::generate(
    //     std::rbegin(v),
    //     std::rend(v),
    //     [t = tokens] () mutable {
    //         auto tmp = t.top();
    //         t.pop();
    //         return tmp;
    //     });

    while (not tokens.empty()) {
        v.insert(v.begin(), tokens.top());
        tokens.pop();
    }

    std::cout << "    ";
    for (auto const& token : v) {
        if (std::holds_alternative<verb>(token)) {
            std::cout << std::get<verb>(token).glyph;
        } else if (std::holds_alternative<adverb>(token)) {
            std::cout << std::get<adverb>(token).glyph;
        } else if (std::holds_alternative<punctuation>(token)) {
            std::cout << (std::get<punctuation>(token).type
                == ParenType::LEFT ? '(' : ')');
        } else if (std::holds_alternative<copula>(token)) {
            std::cout << APLCharSet::LEFT_ARROW;
        } else if (std::holds_alternative<pronoun>(token)) {
            std::cout << std::get<pronoun>(token).name;
        }
        else {
            auto const& n = std::get<noun>(token);
            if (n.type() == noun_type::SCALAR) {
                std::cout << std::get<scalar>(n.data());
            } else if (n.type() == noun_type::VECTOR) {
                std::cout << n;
            } else {
                assert(n.type() == noun_type::NESTED_VECTOR ||
                       n.type() == noun_type::MATRIX);
                std::cout << "...\n\r    " << n << "\n\r";
            }
        }
    }
    std::cout << "\n\r";
}

// monadic
auto evaluate_iota(noun const& n) -> expected_noun {
    if (n.type() == noun_type::SCALAR) {
        auto i = std::get<scalar>(n.data());
        std::vector<int> v(i);
        std::iota(v.begin(), v.end(), 1);
        return noun{v};
    } else {
        return error{"iota not support for rank > 0 yet"};
    }
}

auto evaluate_reverse(noun n) -> expected_noun {
    if (n.type() == noun_type::SCALAR) {
        return n;
    } else if (n.type() == noun_type::VECTOR) {
        // assert(n.rank() == 1);
        auto v = std::get<vector>(n.data());
        std::reverse(v.begin(), v.end());
        return noun{v};
    } else {
        auto v = std::get<nested_vector>(n.data());
        std::reverse(v.begin(), v.end());
        return noun{noun_type::NESTED_VECTOR, v};
    }
}

auto evaluate_unique(noun const& n) -> expected_noun {
    if (n.type() == noun_type::SCALAR) {
        return n;
    } else {
        // TODO this is actually wrong
        // this function should be dedup
        auto v = std::get<vector>(n.data());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v.size() == 1 ? noun{v[0]} : noun{v};
    }
}

auto evaluate_enclose(noun n) -> expected_noun {
    if (n.type() == noun_type::SCALAR) {
        return error{"rank 0 monadic enclose not support yet"};
    } else {
        assert(n.type() == noun_type::VECTOR);
        return noun{noun_type::NESTED_VECTOR, nested_vector{n}};
    }
}

auto evaluate_shape(noun const& n) -> expected_noun {

    std::vector<int> noun_shapes(n.shape().size());

    // TODO once generic - we shouldn't need this transform
    std::transform(
        std::cbegin(n.shape()),
        std::cend(n.shape()),
        std::begin(noun_shapes),
        [] (auto e) { return static_cast<int>(e); });

    return noun_shapes;
}

// TODO easy clean up (similar to dyadic transform)
auto evaluate_absolute_value(noun const& n) -> expected_noun {
    if (n.type() == noun_type::SCALAR) {
        auto i = std::get<scalar>(n.data());
        return std::abs(i);
    } else if (n.type() == noun_type::VECTOR) {
        // TODO can't use const& ?
        auto const v = std::get<vector>(n.data());
        std::vector<int> res(v.size());
        std::transform(
            std::cbegin(v),
            std::cend(v),
            std::begin(res),
            [] (auto const& a) { return std::abs(a); });
        return res;
    } else {
        return error{"absolute value not support for rank > 1 yet"};
    }
}

auto evaluate_not(noun const& n) -> expected_noun {
    if (n.type() == noun_type::SCALAR) {
        auto i = std::get<scalar>(n.data());
        if (i != 0 && i != 1)
            return error{"domain error on not, must be 0 or 1"};
        return 1 - i;
    } else if (n.type() == noun_type::VECTOR) {
        // TODO can't use const& ?
        auto const v = std::get<vector>(n.data());
        std::vector<int> res(v.size());

        auto const domain_error = std::any_of(
            std::cbegin(v),
            std::cend(v),
            [] (auto const& e) { return e != 0 && e != 1; });

        if (domain_error)
            return error{"domain error on not, must be 0 or 1"};

        std::transform(
            std::cbegin(v),
            std::cend(v),
            std::begin(res),
            [] (auto const& e) { return 1 - e; });

        return res;
    } else {
        return error{"not (~) not support for rank > 1 yet"};
    }
}

auto evaluate_sign_of(noun const& n) -> expected_noun {

    auto sign_of = [](auto i) {
        if (i > 0)      return  1;
        else if (i < 0) return -1;
        else            return  0;
    };

    if (n.type() == noun_type::SCALAR) {
        auto i = std::get<scalar>(n.data());
        return sign_of(i);
    } else if (n.type() == noun_type::VECTOR) {
        // TODO can't use const& ?
        auto const v = std::get<vector>(n.data());
        std::vector<int> res(v.size());
        std::transform(
            std::cbegin(v),
            std::cend(v),
            std::begin(res),
            sign_of);
        return res;
    } else {
        return error{"sign_of not support for rank > 1 yet"};
    }
}

auto evalulate_monadic(verb const& mverb,
                       noun const& n) -> expected_noun {
    using namespace APLCharSet;
    if      (mverb.glyph == IOTA)              return evaluate_iota           (n);
    else if (mverb.glyph == REVERSE)           return evaluate_reverse        (n);
    else if (mverb.glyph == UNIQUE)            return evaluate_unique         (n);
    else if (mverb.glyph == ENCLOSE)           return evaluate_enclose        (n);
    else if (mverb.glyph == SHAPE)             return evaluate_shape          (n);
    else if (mverb.glyph == ABSOLUTE_VALUE)    return evaluate_absolute_value (n);
    else if (mverb.glyph == SIGN_OF)           return evaluate_sign_of        (n);
    else if (mverb.glyph == NOT)               return evaluate_not            (n);
    else return error{"monadic " + mverb.glyph + " not supported yet"};
}

// -> x | take m
//      | reverse
//      | max scan

auto evaluate_catenate(noun lhs,
                       noun rhs) -> expected_noun {
    auto lhs_scalar = std::holds_alternative<scalar>(lhs.data());
    auto rhs_scalar = std::holds_alternative<scalar>(rhs.data());
    if (lhs_scalar and rhs_scalar)
        return vector{std::get<scalar>(lhs.data()), std::get<scalar>(rhs.data())};
    else if (lhs_scalar) {
        auto v = std::get<vector>(rhs.data());
        v.insert(v.begin(), std::get<scalar>(lhs.data()));
        return v;
    } else if (rhs_scalar) {
        auto v = std::get<vector>(lhs.data());
        v.push_back(std::get<scalar>(rhs.data()));
        return v;
    } else {
        auto v = std::get<vector>(lhs.data());
        auto u = std::get<vector>(rhs.data());
        v.insert(v.end(), u.begin(), u.end());
        return v;
    }
}

auto evaluate_take(noun const& lhs,
                   noun const& rhs) -> expected_noun {
    if (std::holds_alternative<vector>(lhs.data())) {
        return error{"rank >0 not supported for lhs of take"};
    } else if (std::holds_alternative<scalar>(rhs.data())) {
        return error{"rank 0 not supported for rhs of take"};
    } else {
        auto const l = std::get<scalar>(lhs.data());
        auto const r = std::get<vector>(rhs.data());
        if (l == 1) {
            return r.front();
        }
        return std::vector<int>(r.cbegin(), r.cbegin() + l);
    }
}

auto evaluate_drop(noun const& lhs,
                   noun const& rhs) -> expected_noun {
    if (std::holds_alternative<vector>(lhs.data())) {
        return error{"rank >0 not supported for lhs of drop"};
    } else if (std::holds_alternative<scalar>(rhs.data())) {
        return error{"rank 0 not supported for rhs of drop"};
    } else {
        auto l = std::get<scalar>(lhs.data());
        auto r = std::get<vector>(rhs.data());
        if (l == r.size() - 1) {
            return r.back();
        }
        return std::vector<int>(r.cbegin() + l, r.cend());
    }
}

template <typename BinOp>
auto evaluate_transform_verb(noun const& lhs,
                             noun const& rhs,
                             BinOp binop,
                             std::string verb) -> expected_noun {
    if (lhs.type() == noun_type::SCALAR &&
        rhs.type() == noun_type::SCALAR) {
        auto const l = std::get<scalar>(lhs.data());
        auto const r = std::get<scalar>(rhs.data());
        return binop(l, r);
    } else if (lhs.type() == noun_type::SCALAR &&
               rhs.type() == noun_type::VECTOR) {
        // TODO investigate why `r` can't be a reference
        auto const l = std::get<scalar>(lhs.data());
        auto const r = std::get<vector>(rhs.data());
        std::vector<int> res(r.size());

        std::transform(
            std::cbegin(r),
            std::cend(r),
            std::begin(res),
            std::bind(binop, l, std::placeholders::_1));

        return res;
    } else if (lhs.type() == noun_type::VECTOR &&
               rhs.type() == noun_type::SCALAR) {
        // TODO investigate why `r` can't be a reference
        auto const l = std::get<vector>(lhs.data());
        auto const r = std::get<scalar>(rhs.data());
        std::vector<int> res(l.size());

        std::transform(
            std::cbegin(l),
            std::cend(l),
            std::begin(res),
            std::bind(binop, std::placeholders::_1, r));

        return res;
    } else if (lhs.type() == noun_type::VECTOR &&
             rhs.type() == noun_type::VECTOR) {
        // TODO investigate why `l` and `r` can't be a reference
        auto const l = std::get<vector>(lhs.data());
        auto const r = std::get<vector>(rhs.data());
        std::vector<int> res(l.size());

        std::transform(
            std::cbegin(l),
            std::cend(l),
            std::cbegin(r),
            std::begin(res),
            binop);

        return res;
    } else {
        return error{"ranks of lhs/rhs not supported for"s + verb};
    }
}

auto evaluate_equal_to(noun const& lhs, noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs, std::equal_to{}, "equal_to"s);
}

auto evaluate_not_equal_to(noun const& lhs, noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs, std::not_equal_to{}, "not_equal_to"s);
}

auto evaluate_residue(noun const& lhs,
                      noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs,
        [](auto const &a, auto const& b) { return b % a; },
        "residue"s);
}

auto evaluate_subtract(noun const& lhs, noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs, std::minus{}, "subtract"s);
}

auto evaluate_add(noun const& lhs, noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs, std::plus{}, "add"s);
}

auto evaluate_multiply(noun const& lhs, noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs, std::multiplies{}, "mulitply"s);
}

auto evaluate_less_than(noun const& lhs, noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs, std::less{}, "less than"s);
}

auto evaluate_less_than_or_equal(noun const& lhs, noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs, std::less_equal{}, "less than or equal"s);
}

auto evaluate_greater_than(noun const& lhs, noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs, std::greater{}, "greater than"s);
}

auto evaluate_greater_than_or_equal(noun const& lhs, noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs, std::greater_equal{}, "greater than or equal"s);
}

auto evaluate_maximum(noun const& lhs, noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs,
        [](auto const& a, auto const& b) { return std::max(a, b); },
        "maximum"s);
}

auto evaluate_minimum(noun const& lhs, noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs,
        [](auto const& a, auto const& b) { return std::min(a, b); },
        "minimum"s);
}

auto evaluate_power(noun const& lhs, noun const& rhs) -> expected_noun {
    return evaluate_transform_verb(lhs, rhs,
        [](auto const& a, auto const& b) { return std::pow(a, b); },
        "power"s);
}

auto evaluate_rotate(noun const& lhs,
                     noun rhs) -> expected_noun {
    if (lhs.type() == noun_type::SCALAR
     && rhs.type() == noun_type::VECTOR) {
        auto const val = std::get<scalar>(lhs.data());
        auto       v   = std::get<vector>(rhs.data());

        std::rotate(
            std::begin(v),
            std::begin(v) + val,
            std::end(v));

        return v;
    } else {
        return error{"ranks of lhs/rhs not supported for rotate"};
    }
}

auto evaluate_match(noun const& lhs,
                    noun const& rhs) -> expected_noun {
    if (lhs.type() == noun_type::SCALAR
     && rhs.type() == noun_type::SCALAR) {
        auto const l = std::get<scalar>(lhs.data());
        auto const r = std::get<scalar>(rhs.data());
        return l == r;
    } else if (lhs.type() == noun_type::VECTOR
            && rhs.type() == noun_type::VECTOR) {
        auto const& l = std::get<vector>(lhs.data());
        auto const& r = std::get<vector>(rhs.data());

        return std::equal(
            std::begin(l),
            std::end(l),
            std::begin(r));

    } else if (lhs.type() == noun_type::SCALAR
            || rhs.type() == noun_type::SCALAR) {
        return 0;
    } else {
        return error{"ranks >1 not supported for match"};
    }
}

auto evaluate_replicate(noun const& lhs,
                        noun const& rhs) -> expected_noun {
    if (lhs.type() == noun_type::SCALAR
     && rhs.type() == noun_type::SCALAR) {
        auto const times = std::get<scalar>(lhs.data());
        auto const val   = std::get<scalar>(rhs.data());
        return vector(times, val);
    } else if (lhs.type() == noun_type::SCALAR
            && rhs.type() == noun_type::VECTOR) {
        auto const val = std::get<scalar>(lhs.data());
        auto const v   = std::get<vector>(rhs.data());

        return std::accumulate(
            std::cbegin(v),
            std::cend(v),
            vector{},
            [val] (auto acc, auto e) {
                acc.resize(acc.size() + val, e);
                return acc;
            });

    } else if (lhs.type() == noun_type::VECTOR
            && rhs.type() == noun_type::VECTOR) {
        auto const l = std::get<vector>(lhs.data());
        auto const r = std::get<vector>(rhs.data());

        return std::inner_product(
            std::cbegin(l),
            std::cend(l),
            std::cbegin(r),
            vector{},
            [](auto& acc, auto e) {
                acc.insert(acc.end(), e.begin(), e.end());
                return acc;
            },
            [](auto times, auto val) {
                return vector(times, val);
            });

    } else {
        return error{"ranks not supported for replicate"};
    }
}

auto evaluate_partitioned_enclose(noun const& lhs,
                                  noun const& rhs) -> expected_noun {
    if (lhs.type() == noun_type::SCALAR) {
        auto const times = std::get<scalar>(lhs.data());
        if (times != 0 && times != 1) {
            return error{"Domain error: lhs scalar has to be 0 or 1"};
        } else {
            return error {"Not implemented"};
        }
        // auto const val   = std::get<scalar>(rhs.data());
        // return vector(times, val);
    } else if (lhs.type() == noun_type::VECTOR
            && rhs.type() == noun_type::VECTOR) {
        auto const l = std::get<vector>(lhs.data());
        auto const r = std::get<vector>(rhs.data());

        if (l.size() != r.size()) {
            return error{"Size of lhs and rhs don't equal"};
        }

        // lol, wtf - C++ so weak compared to Haskell
        // partitioned_enclose :: [Bool] -> [a] -> [[a]]
        // partitioned_enclose = map (map snd)
        //                     . tail
        //                     . segmentBefore ((==1) . fst)
        //                     . zip

        auto [nv, last] = std::inner_product(
            std::cbegin(l),
            std::cend(l),
            std::cbegin(r),
            // TODO need to make the int generic
            std::pair{nested_vector{}, std::vector<int>{}},
            [first = true] (auto& acc, auto p) mutable {
                auto  [mask, val] = p;
                auto& [nv,   v]   = acc;
                if (mask) {
                    if (not first) nv.push_back(noun(v));
                    else first = false;
                    return std::make_pair(nv, std::vector{val});
                } else {
                    if (not first) v.push_back(val);
                    return std::make_pair(nv, v);
                }
            },
            [](auto mask, auto val) {
                return std::make_pair(mask, val);
            });
        nv.push_back(last);
        return noun{noun_type::NESTED_VECTOR, nv};

    } else {
        return error{"ranks not supported for partitioned close"};
    }
}

auto evaluate_partition(noun const& lhs,
                        noun const& rhs) -> expected_noun {
    if (lhs.type() == noun_type::SCALAR) {
        // auto const times = std::get<scalar>(lhs.data());
        return error {"Not implemented"};
    } else if (lhs.type() == noun_type::VECTOR
            && rhs.type() == noun_type::VECTOR) {
        auto const l = std::get<vector>(lhs.data());
        auto const r = std::get<vector>(rhs.data());

        if (l.size() != r.size()) {
            return error{"Size of lhs and rhs don't equal"};
        }

        auto [nv, last] = std::inner_product(
            std::cbegin(l),
            std::cend(l),
            std::cbegin(r),
            // TODO need to make the int generic
            std::pair{nested_vector{}, std::vector<int>{}},
            [] (auto& acc, auto p) {
                auto  [mask, val] = p;
                auto& [nv,   v]   = acc;
                if (mask) {
                    v.push_back(val);
                    return std::make_pair(nv, v);
                } else {
                    if (not v.empty()) nv.push_back(v);
                    return std::make_pair(nv, std::vector<int>{});
                }
            },
            [](auto mask, auto val) {
                return std::make_pair(mask, val);
            });
        if (not last.empty()) nv.push_back(last);
        return noun{noun_type::NESTED_VECTOR, nv};

    } else {
        return error{"ranks not supported for partitioned close"};
    }
}

auto evaluate_reshape(noun const& lhs,
                      noun const& rhs) -> expected_noun {
    if (lhs.type() == noun_type::SCALAR) {
        auto const l = std::get<scalar>(lhs.data());
        if (rhs.type() == noun_type::SCALAR) {
            auto const r = std::get<scalar>(rhs.data());
            return std::vector(l, r);
        } else if (rhs.type() == noun_type::VECTOR) {
            auto const r = std::get<vector>(rhs.data());
            if (l<= r.size()) {
                return vector(r.begin(), r.begin() + l);
            } else {
                auto res = r;
                while (res.size() < l) {
                    auto end = std::min(res.size(), l - res.size());
                    res.insert(res.end(), r.begin(), r.begin() + end);
                }
                return res;
            }
        } else { // MATRIX
            // TODO
        }
    } else if (lhs.type() == noun_type::VECTOR) {
        // std::cout << to_string(lhs) << '\n';
        // std::cout << to_string(lhs.shape()) << '\n';
        assert(lhs.shape().size() == 1 && lhs.shape().front() == 2);
        auto const l    = std::get<vector>(lhs.data());
        auto const rows = l.front();
        auto const cols = l.back();
        if (rhs.type() == noun_type::SCALAR) {
            auto const r = std::get<scalar>(rhs.data());
            return std::vector(rows, std::vector(cols, r));
        } else if (rhs.type() == noun_type::VECTOR) {
            auto const r = std::get<vector>(rhs.data());
            std::vector res(rows, std::vector(cols, 0));
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j)
                    res[i][j] = r[(i * cols + j) % r.size()];
            }
            return res;
        } else { // MATRIX
            // TODO
        }
    }
    return error{"ranks not supported for reshape"};
}

// template <typename T>
auto evaluate_dyadic(noun const& lhs,
                     verb const& dverb,
                     noun const& rhs) -> expected_noun {
    using namespace APLCharSet;
    if      (dverb.glyph == CATENATE)              return evaluate_catenate              (lhs, rhs);
    else if (dverb.glyph == TAKE)                  return evaluate_take                  (lhs, rhs);
    else if (dverb.glyph == DROP)                  return evaluate_drop                  (lhs, rhs);
    else if (dverb.glyph == EQUAL_TO)              return evaluate_equal_to              (lhs, rhs);
    else if (dverb.glyph == NOT_EQUAL_TO)          return evaluate_not_equal_to          (lhs, rhs);
    else if (dverb.glyph == ROTATE)                return evaluate_rotate                (lhs, rhs);
    else if (dverb.glyph == MATCH)                 return evaluate_match                 (lhs, rhs);
    else if (dverb.glyph == REPLICATE)             return evaluate_replicate             (lhs, rhs);
    else if (dverb.glyph == RESIDUE)               return evaluate_residue               (lhs, rhs);
    else if (dverb.glyph == PARTITIONED_ENCLOSE)   return evaluate_partitioned_enclose   (lhs, rhs);
    else if (dverb.glyph == PARTITION)             return evaluate_partition             (lhs, rhs);
    else if (dverb.glyph == SUBTRACT)              return evaluate_subtract              (lhs, rhs);
    else if (dverb.glyph == ADD)                   return evaluate_add                   (lhs, rhs);
    else if (dverb.glyph == MULTIPLY)              return evaluate_multiply              (lhs, rhs);
    else if (dverb.glyph == MAXIMUM)               return evaluate_maximum               (lhs, rhs);
    else if (dverb.glyph == MINIMUM)               return evaluate_minimum               (lhs, rhs);
    else if (dverb.glyph == RESHAPE)               return evaluate_reshape               (lhs, rhs);
    else if (dverb.glyph == POWER)                 return evaluate_power                 (lhs, rhs);
    else if (dverb.glyph == LESS_THAN)             return evaluate_less_than             (lhs, rhs);
    else if (dverb.glyph == LESS_THAN_OR_EQUAL)    return evaluate_less_than_or_equal    (lhs, rhs);
    else if (dverb.glyph == GREATER_THAN)          return evaluate_greater_than          (lhs, rhs);
    else if (dverb.glyph == GREATER_THAN_OR_EQUAL) return evaluate_greater_than_or_equal (lhs, rhs);

    else return error{"dyadic " + dverb.glyph + " not supported yet"};
}

auto is_composable_with_binary_op_adverb(std::string_view s) -> bool {

    auto const MAX = APLCharSet::MAXIMUM;
    auto const MIN = APLCharSet::MINIMUM;
    auto const POW = APLCharSet::POWER;

    return s == "+" || s == "-" ||
           s == "×" || s == "÷" ||
           s == "∧" || s == "∨" ||
           s == "<" || s == ">" ||
           s == "≤" || s == "≥" ||
           s == MIN || s == MAX ||
           s == POW;
}

template <typename T>
auto get_binop(verb const& cverb) -> std::function<T(T, T)> {

    auto const MAX = APLCharSet::MAXIMUM;
    auto const MIN = APLCharSet::MINIMUM;

    if      (cverb.glyph == "+") return std::plus          <T>();
    else if (cverb.glyph == "<") return std::less          <T>();
    else if (cverb.glyph == "-") return std::minus         <T>();
    else if (cverb.glyph == ">") return std::greater       <T>();
    else if (cverb.glyph == "÷") return std::divides       <T>();
    else if (cverb.glyph == "≤") return std::less_equal    <T>();
    else if (cverb.glyph == "×") return std::multiplies    <T>();
    else if (cverb.glyph == "∨") return std::logical_or    <T>();
    else if (cverb.glyph == "∧") return std::logical_and   <T>();
    else if (cverb.glyph == "≥") return std::greater_equal <T>();
    else if (cverb.glyph == MAX) return [](auto const& a, auto const& b) { return std::max(a, b); };
    else if (cverb.glyph == MIN) return [](auto const& a, auto const& b) { return std::min(a, b); };

    return std::plus<T>();
}

template <typename B, typename Binop>
auto fold_right(B f, B l, Binop binop) {
    if (std::distance(f, l) == 1) return *f;
    auto acc = *--l;
    for (;;) {
        acc = binop(*--l, acc);
        if (f == l) break;
    }
    return acc;
}

template <typename I, typename O, typename Binop>
auto scan_left_wtf(I f, I l, O o, Binop binop) {
    *o = *f;
    for (auto e = f + 2; e <= l; ++e)
        *++o = fold_right(f, e, binop);
}

auto evaluate_reduce(verb const& lhs,
                     noun const& rhs) -> expected_noun {
    assert(is_composable_with_binary_op_adverb(lhs.glyph));
    if (rhs.type() == noun_type::VECTOR) {
        auto const v = std::get<vector>(rhs.data());

        return fold_right(
            std::cbegin(v),
            std::cend(v),
            get_binop<int/*decltype(*std::begin(v))*/>(lhs));

    } else if (rhs.type() == noun_type::MATRIX) {
        auto const m = std::get<matrix>(rhs.data());
        std::vector<int> res(rhs.shape().front()); // num of rows = len of returned vector

        std::transform(
            std::cbegin(m),
            std::cend(m),
            std::begin(res),
            [lhs] (auto const& row) { return std::get<scalar>(
                                        evaluate_reduce(lhs, noun{row})
                                                .value()
                                                .data()); });

        return res;
    } else return error{"rank " + std::to_string(rhs.rank())
                      + " not supported for reduce adverb"};
}

auto evaluate_scan(verb const& lhs,
                   noun const& rhs) -> expected_noun {
    assert(is_composable_with_binary_op_adverb(lhs.glyph));
    if (rhs.type() == noun_type::VECTOR) {
        auto const v = std::get<vector>(rhs.data());
        auto res = v;

        scan_left_wtf(
            std::cbegin(v),
            std::cend(v),
            std::begin(res),
            get_binop<int/*decltype(*std::begin(v))*/>(lhs));

        return res;

    } else return error{"rank " + std::to_string(rhs.rank())
                      + " not supported for reduce adverb"};
}

auto eval(std::stack<token> tokens, bool first) -> noun;

void resolve_parens(std::stack<token>& tokens) {
    std::vector<token> tmp;
    int paren_count = 0;
    do {
        tmp.push_back(tokens.top());
        tokens.pop();
        if (std::holds_alternative<punctuation>(tmp.back())) {
            auto token = std::get<punctuation>(tmp.back());
            if      (token.type == ParenType::RIGHT) ++paren_count;
            else if (token.type == ParenType::LEFT) --paren_count;
        }
    } while(paren_count > 0);
    tmp.pop_back();
    tmp = std::vector(tmp.rbegin(), tmp.rend());
    tmp.pop_back();
    std::stack<token> expr;
    for (auto e : tmp) expr.push(e);
    auto new_expr = eval(expr, false);
    tokens.push(new_expr);
}

auto get_noun(token const& t) -> noun {
    return std::holds_alternative<noun>(t)
        ? std::get<noun>(t)
        : pronoun_list.find(std::get<pronoun>(t).name)->second;
}

auto eval(std::stack<token> tokens, bool first_level) -> noun {

    while (not tokens.empty()) {

        // rhs parens
        if (std::holds_alternative<punctuation>(tokens.top())) {
            auto const& paren = std::get<punctuation>(tokens.top());
            assert(paren.type == ParenType::RIGHT && "LHS paren isn't actually a paren");
            resolve_parens(tokens);
        }

        if (tokens.size() > 1) {
            if (first_level)
                print_flat_tokens(tokens);
        } else {
            if (first_level)
                std::cout << "    ";
            return get_noun(tokens.top());
        }

        auto rhs = get_noun(tokens.top());
        tokens.pop();
        if (tokens.empty()) {
            std::cout << rhs; // maybe create print_noun
            break;
        }

        // left of a noun should always be on of:
        // copula, verb OR adverb
        assert(std::holds_alternative<verb>  (tokens.top()) ||
               std::holds_alternative<adverb>(tokens.top()) ||
               std::holds_alternative<copula>(tokens.top()));

        if (std::holds_alternative<copula>(tokens.top())) {
            // process ←
            tokens.pop();
            assert(not tokens.empty());
            assert(std::holds_alternative<pronoun>(tokens.top()));
            auto const var = std::get<pronoun>(tokens.top());
            tokens.pop();
            pronoun_list.insert({var.name, rhs});
            tokens.push(rhs);
        } else if (std::holds_alternative<adverb>(tokens.top())) {
            auto adv = std::get<adverb>(tokens.top());
            tokens.pop();
            auto const& lhs = std::get<verb>(tokens.top());
            assert(is_composable_with_binary_op_adverb(lhs.glyph));

            auto exp_new_subj = [&] {
                if (adv.glyph == "/") {
                    return evaluate_reduce(lhs, rhs);
                } else {
                    assert(adv.glyph == "\\");
                    return evaluate_scan(lhs, rhs);
                }
            } ();

            if (not exp_new_subj.has_value()) {
                std::cout << COLOR_ERROR <<exp_new_subj.error();
                return noun{0};
            }

            tokens.pop();
            tokens.push(exp_new_subj.value());
        } else {
            assert(std::holds_alternative<verb>(tokens.top()));
            auto const v = std::get<verb>(tokens.top());
            tokens.pop();

            if (tokens.empty() || std::holds_alternative<verb>  (tokens.top())
                               || std::holds_alternative<adverb>(tokens.top())
                               || std::holds_alternative<copula>(tokens.top())) {
                auto const& mverb = v;
                auto exp_new_subj = evalulate_monadic(mverb, rhs);
                if (not exp_new_subj.has_value()) {
                    std::cout << COLOR_ERROR <<exp_new_subj.error();
                    return noun{0};
                }
                tokens.push(exp_new_subj.value());
            } else {

                auto const& dverb = v;

                if (std::holds_alternative<punctuation>(tokens.top())) {
                    // duplicated code from above
                    auto const& paren = std::get<punctuation>(tokens.top());
                    assert(paren.type == ParenType::RIGHT);
                    resolve_parens(tokens);
                }

                assert(std::holds_alternative<noun>   (tokens.top()) ||
                       std::holds_alternative<pronoun>(tokens.top()));

                auto const& lhs = get_noun(tokens.top());
                auto exp_new_subj = evaluate_dyadic(lhs, dverb, rhs);
                if (not exp_new_subj.has_value()) {
                    std::cout << COLOR_ERROR << exp_new_subj.error();
                    return noun{0};
                }
                tokens.pop();
                tokens.push(exp_new_subj.value());
            }
        }
    }
    return noun{0};
}

enum class UnitTest {
    PASS,
    FAIL
};

struct unit_test_result {
    std::string test;
    UnitTest result;
};

std::ostream& operator<<(std::ostream& os, unit_test_result const& res) {
    if (res.result == UnitTest::PASS) {
        os << termcolor::green << "PASS";
    } else {
        os << termcolor::red   << "FAIL";
    }
    os << " -> " << res.test;
    return os;
}

auto unit_test(std::string test, std::string expected) -> unit_test_result {
    auto const tokens = tokenize(test);
    auto const n      = eval(tokens, false);
    auto const res    = to_string(n.data());
    // std::cout << res << " == " << expected << '\n'; // DEBUGGING
    return unit_test_result { test, res == expected ? UnitTest::PASS
                                                    : UnitTest::FAIL };
}

void run_tests() {

    noun b(9);
    std::cout << termcolor::yellow << to_string(b) << "\n\n";

    noun a({1, 2, 3});
    std::cout << to_string(a) << "\n\n";

    noun c({{1, 2, 3, 4}, {3, 4, 5, 6}, {5, 6, 7, 8}});
    std::cout << to_string(c) << "\n\n";

    auto x = nested_vector{};
    std::cout << to_string(x) << "\n\n";

    std::cout << unit_test("⍳5",               "1 2 3 4 5") << "\n\r"
              << unit_test("⌽⍳5",             "5 4 3 2 1") << "\n\r"
              << unit_test("∪1 1 2 2 3 3",    "1 2 3")     << "\n\r"
              << unit_test("⍳(∪3 3 3)",        "1 2 3")    << "\n\r"
              << unit_test("3↑⌽⍳5",           "5 4 3")     << "\n\r"
              << unit_test("⌽(⌽3↑⍳5),⌽3↓⍳5", "4 5 1 2 3") << "\n\r"
              << unit_test("3⌽⍳5",            "4 5 1 2 3") << "\n\r"
              << unit_test("5/3",             "3 3 3 3 3") << "\n\r"
              << unit_test("(⍳3)≡⌽⌽⍳3",       "1")        << "\n\r"
              << unit_test("(0≠2|⍳5)/⍳5",      "1 3 5")     << "\n\r"
              << unit_test("∪0≠2|⍳5",         "1 0")       << "\n\r"
              << unit_test("((3/1),2)≠4/1",   "0 0 0 1")   << "\n\r"
              << unit_test("(⍳4)|4/2",         "0 0 2 2")   << "\n\r"
              << unit_test("8-4",             "4")         << "\n\r"
              << unit_test("4,4+(×8-4)×⍳|8-4", "4 5 6 7 8") << "\n\r"   // TO idiom
              << unit_test("1≠(2/1),1+⍳2",     "0 0 1 1")   << "\n\r"
              << unit_test("((2/1),1+⍳2)≠1",   "0 0 1 1")   << "\n\r"
              << unit_test("(∨\\(⍳3)≠1)/⍳3",    "2 3")       << "\n\r"  // LTRIM idiom
              << unit_test("⌈/⍳5",              "5")         << "\n\r"
              << unit_test("⌊\\2⌽2×⍳5",        "6 6 6 2 2") << "\n\r"
              << unit_test("+/2×(0=2|⍳20)/⍳+/10 10", "220") << "\n\r"
              << unit_test("m←5",              "5")        << "\n\r"
              << unit_test("+/(2|x)/x←⍳10",     "25")       << "\n\r"
              << unit_test("+/(~2|x)/x←⍳10",    "30")       << "\n\r"
              << unit_test("|(⍳4)-2",           "1 0 1 2")  << "\n\r"
              << unit_test("×(⍳4)-2",           "-1 0 1 1") << "\n\r"
              << unit_test("-/⍳9",              "5")        << "\n\r"
              << unit_test("-\\⍳5",          "1 -1 2 -2 3") << "\n\r"
              << unit_test("1↑3↓+/7 7⍴10*(⍳4)-1", "2221")   << "\n\r"
              << unit_test(">\\⌽⍳4",           "4 1 1 1")  << "\n\r";
            //   << unit_test("+/0>,2 2⍴(⍳4)-3",   "2")        << "\n\r";

              // ⍴∘⍴¨x ← 'abc' 123 (3 3⍴⍳9)
              // (1,2>/x)⊂x ← (4⌽⍳9),2⌽⍳6  // N-Wise Reduction
}

std::string padRight(std::string const& input, char add, int num) {
    auto tmp = input;
    tmp.resize(num, add);
    return tmp;
}

void print_dyadic_supported_characters() {
    using namespace APLCharSet;
    int const width = 25;
    std::cout << CATENATE            << padRight(" CATENATE",     ' ' , width);
    std::cout << TAKE                << padRight(" TAKE",         ' ' , width);
    std::cout << DROP                << padRight(" DROP",         ' ' , width) << "\n\r";
    std::cout << EQUAL_TO            << padRight(" EQUAL_TO",     ' ' , width);
    std::cout << NOT_EQUAL_TO        << padRight(" NOT_EQUAL_TO", ' ' , width);
    std::cout << ROTATE              << padRight(" ROTATE",       ' ' , width) << "\n\r";
    std::cout << MATCH               << padRight(" MATCH",        ' ' , width);
    std::cout << REPLICATE           << padRight(" REPLICATE",    ' ' , width);
    std::cout << RESIDUE             << padRight(" RESIDUE",      ' ' , width) << "\n\r";
    std::cout << PARTITION           << padRight(" PARTITION",    ' ' , width);
    std::cout << PARTITIONED_ENCLOSE << padRight(" PARTITIONED_ENCLOSE", ' ' , width) << "\n\r";
}

int main() {

    std::cout << COLOR_HIGHLIGHT << "\n === AHI ===\n\n";
    // AHI = APL Hoekstra Interpreter

    run_tests();
    std::cout << "\n";

    int c;
    // use system call to make terminal send all keystrokes directly to stdin
    system ("/bin/stty raw");

    // initial ncurses
    // initscr();
    // raw();
    // keypad(stdscr, TRUE);
    // noecho();

    bool aplCharIncoming   = false;
    bool aplStringIncoming = false;
    std::string input;
    // std::string prev_input = "";
    std::string aplString;
    std::cout << COLOR_HIGHLIGHT << "> " << COLOR_INPUT;
    while((c=getchar()) != '#') {
        if (c == '`')
            aplCharIncoming = true;
        else if (aplCharIncoming) {
            aplCharIncoming = false;
            putchar(ASCII::BACKSPACE);
            putchar(ASCII::BACKSPACE);
            auto const aplChar = getAplCharFromShortCut(c);
            std::cout << aplChar;
            input += aplChar;
            putchar(' ');
            putchar(ASCII::BACKSPACE);
        } else if (c == '') { // backspace
            putchar(ASCII::BACKSPACE);
            putchar(ASCII::BACKSPACE);
            std::cout << "  ";
            putchar(ASCII::BACKSPACE);
            putchar(ASCII::BACKSPACE);
            if (aplStringIncoming) {
                aplString.pop_back();
            } else {
                input.pop_back();
            }
        } else if (c == '\'') {
            aplStringIncoming = true;
        } else if (aplStringIncoming) {
            if (c == ' ') {
                auto const aplChar = getAplCharFromString(aplString);
                for (int i = 0; i < aplString.size() + 2; ++i) {
                    putchar(ASCII::BACKSPACE);
                    putchar(' ');
                    putchar(ASCII::BACKSPACE);
                }
                std::cout << aplChar;
                input += aplChar;
                aplStringIncoming = false;
                aplString.clear();
            } else {
                aplString += c;
            }
        } else if (c == ASCII::RETURN) {
            putchar(ASCII::BACKSPACE);
            putchar(ASCII::BACKSPACE);
            std::cout << "  \n\r";
            std::cout << COLOR_OUTPUT;
            // prev_input = input;
            if (input == "]TEST") {
                run_tests();
            } else if (input == "]MONADIC") {
                // print_monadic_supported_characters();
            } else if (input == "]DYADIC") {
                print_dyadic_supported_characters();
            } else {
                auto tokens = tokenize(input);
                auto nn = eval(tokens, true);
                std::cout << nn << "\n\r";;
            }
            std::cout << COLOR_HIGHLIGHT << "\r> " << COLOR_INPUT;
            input.clear();
        // } else if (c == KEY_UP) {
        //     input = prev_input;
        //     putchar(ASCII::BACKSPACE);
        //     putchar(ASCII::BACKSPACE);
        //     putchar(ASCII::BACKSPACE);
        //     std::cout << input;
        } else {
            input += c;
            // std::cout << "hello";
        }
    }

    /* use system call to set terminal behaviour to more normal behaviour */
    system ("/bin/stty cooked");
    std::cout << '\n';
    return 0;
}