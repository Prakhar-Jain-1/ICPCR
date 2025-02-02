/*-------------------------------------------------------------------------*
 |                                                                         
 |            STATA SUPPLEMENTAL SYNTAX FILE FOR ICPSR 36151
 |          INDIA HUMAN DEVELOPMENT SURVEY-II (IHDS-II), 2011-12
 |                   (DATASET 0009: SCHOOL FACILITIES)
 |
 |
 | This Stata missing value recode program is provided for optional use with
 | the Stata system version of this data file as distributed by ICPSR.
 | The program replaces user-defined numeric missing values (e.g., -9)
 | with generic system missing "."  Note that Stata allows you to specify
 | up to 27 unique missing value codes.  Only variables with user-defined
 | missing values are included in this program.
 |
 | To apply the missing value recodes, users need to first open the
 | Stata data file on their system, apply the missing value recodes if
 | desired, then save a new copy of the data file with the missing values
 | applied.  Users are strongly advised to use a different filename when
 | saving the new file.
 |
 *------------------------------------------------------------------------*/

replace SC14A = . if (SC14A == 88)
replace PS10C = . if (PS10C == 3)
replace PS24C = . if (PS24C == 88)
replace SO34A = . if (SO34A == 8 | SO34A == 9)
replace SO34B = . if (SO34B == 8 | SO34B == 9)
replace SO34C = . if (SO34C == 8 | SO34C == 9)
replace SO34D = . if (SO34D == 8 | SO34D == 9)
replace SO34E = . if (SO34E == 8 | SO34E == 9)
replace SO35 = . if (SO35 == 8 | SO35 == 9)
replace SO42A = . if (SO42A == 9)
replace SO42B = . if (SO42B == 9)
replace SO42C = . if (SO42C == 9)


