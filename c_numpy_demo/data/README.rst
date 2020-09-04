.. README.rst for data files

Options data
============

This ``README.rst`` gives some information on the CSV data files included in
this package, which contain data on Eurodollar options used in the test suite
and in the benchmark tests [#]_.

.. [#] This benchmark test has yet to be written.

Format
------

The CSV files have the following headers, which are described below.

ccode
   The Bloomberg contract code with 2-digit year for the option, ex. EDU21.

opt_price
   The price of the option, which for Eurodollar options is in units of
   `IMM index points`__, which have the same units as percentage.

fut_price
   The price of the options' underlying, in this case, a futures contract, which
   is also quoted in IMM index points.

strike
   The option's strike, which is also in IMM index points.

dfactor
   The option's discount factor. Note that these are approximate ballpark
   estimates based off of my own experience, as I do not have a model for
   calibrating discount curves. Discount factors are typically more important
   when it comes to longer-dated options, however.

call_put
   +/-1 call/put indicator, where +1 for a call, -1 for a put.

opt_exp
   The option's yyyy-mm-dd expiration date.

fut_exp
   The underlying's, here the future contract's, yyyy-mm-dd expiration date.
   Although this field is not used, it is a useful bookkeeping field as some of
   the options used are Eurodollar mid-curve options, which expiring forward of
   the futures contract they are written on.

rec_date
   Date that the data for this traded option was recorded. Used to compute time
   to maturity.

.. __: https://www.cmegroup.com/education/courses/introduction-to-eurodollars/
   understanding-imm-price-and-date.html

Data files
----------

We provide some brief descriptions on the contents of the data files.

edo_ntm_data.csv
   Contains data on Eurodollar options that are close to the at-the-money point.
   These values were record on Aug 31, 2020, with the specific options contract
   in question 2EH21, the two-year mid-curve Eurodollar option expiring on March
   12, 2021. The underlying futures expires on March 13, 2023. This file
   contains 20 rows and 9 columns of data.

edo_full_data.csv
   Contains data on Eurodollar options on a wider range of strikes and is a
   superset of the data in ``edo_atm_data.csv``. Contains additional data on
   Eurodollar options recorded on September 3, 2020, with the specific contract
   being the 0EU21 contract, the one-year mid-curve Eurodollar option expiring
   on September 10, 2021. The underlying futures expires on September 19, 2022.
   Note that the extremely deep in-the-money options will inevitably end up with
   nonsensical implied volatility values since they are often stale. The file
   contains 136 rows and 9 columns of data.