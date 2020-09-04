.. README.rst for data files

Options data
============

This ``README.rst`` gives some information on the CSV data files included in
this package used in the test suite and in the benchmark tests [#]_.

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
   The option's ``yyyy-mm-dd`` expiration date.

fut_exp
   The underlying's, here the future contract's, ``yyyy-mm-dd`` expiration date.
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

Both data files contain options data pulled from CME Group's website. Brief
descriptions are below.

edo_ntm_data.csv
   Contains data on Eurodollar options that are close to the at-the-money point,
   namely the five closest strikes above and below the at-the-money point. This
   file contains 80 rows and 9 columns of data.

edo_full_data.csv
   Contains data on Eurodollar options on a much wider range of strikes and is a
   superset of the data in ``edo_atm_data.csv``. Note that extremely deep
   in-the-money options will end up with nonsensical implied volatilities.
   The file contains 422 rows and 9 columns of data.

Options contracts used
----------------------

Below are the listed options contracts used in the data files described above.

2EH21
   Two-year mid-curve Eurodollar option expiring on March 12, 2021. Underlying
   futures expires on March 13, 2023.

0EU21
   One-year mid-curve Eurodollar option expiring on September 10, 2021.
   Underlying futures expires on September 19, 2022.

3EM21
   Three-year mid-curve Eurodollar option expiring on Jun 11, 2021. Underlying
   futures expires on June 17, 2024.

EDZ22
   Quarterly Eurodollar option expiring on December 19, 2022. Underlying futures
   expires on the same date.