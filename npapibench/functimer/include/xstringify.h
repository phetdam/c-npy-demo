/**
 * @file xstringify.h
 * @brief Provides macros that turn a list of arguments into a string.
 */

#ifndef FUNCTIMER_XSTRINGIFY_H
#define FUNCTIMER_XSTRINGIFY_H

// stringify combines varargs into a string, i.e. a, b, c becomes "a, b, c",
// while xstringify adds another level of indirection to allow macro expansion.
#define stringify(...) #__VA_ARGS__
#define xstringify(...) stringify(__VA_ARGS__)

#endif /* FUNCTIMER_XSTRINGIFY_H */