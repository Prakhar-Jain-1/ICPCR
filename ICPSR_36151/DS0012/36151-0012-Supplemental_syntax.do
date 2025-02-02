/*-------------------------------------------------------------------------*
 |                                                                         
 |            STATA SUPPLEMENTAL SYNTAX FILE FOR ICPSR 36151
 |          INDIA HUMAN DEVELOPMENT SURVEY-II (IHDS-II), 2011-12
 |                        (DATASET 0012: VILLAGE)
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

replace VG14A = . if (VG14A == 9)
replace VG15A = . if (VG15A == 9)
replace VG16A = . if (VG16A == 9)
replace VG17A = . if (VG17A == 9)
replace VG17E = . if (VG17E == 8)
replace VG18A = . if (VG18A == 9)
replace VG18E = . if (VG18E == 8)
replace VG19A = . if (VG19A == 9)
replace VG19E = . if (VG19E == 8)
replace VG20A = . if (VG20A == 9)
replace VG21A = . if (VG21A == 9)
replace VG22A = . if (VG22A == 9)
replace VG23A = . if (VG23A == 9)
replace VG24A = . if (VG24A == 9)
replace VI12A = . if (VI12A == 9)
replace VI12B = . if (VI12B == 9)
replace VI17 = . if (VI17 == 99)
replace VSE5 = . if (VSE5 == 888)
replace VSE6 = . if (VSE6 == 888)
replace VSE7 = . if (VSE7 == 888)
replace VSE8 = . if (VSE8 == 888)
replace VSE9 = . if (VSE9 == 888)
replace VSE10 = . if (VSE10 == 888)
replace VMB1 = . if (VMB1 == 99)
replace VMC1 = . if (VMC1 == 9)
replace VMB2 = . if (VMB2 == 99)
replace VMC2 = . if (VMC2 == 9)
replace VMB3 = . if (VMB3 == 99)
replace VMC3 = . if (VMC3 == 9)
replace VMB4 = . if (VMB4 == 99)
replace VMC4 = . if (VMC4 == 9)
replace VMB5 = . if (VMB5 == 99)
replace VMC5 = . if (VMC5 == 9)
replace VMB6 = . if (VMB6 == 99)
replace VMC6 = . if (VMC6 == 9)
replace VMB7 = . if (VMB7 == 99)
replace VMC7 = . if (VMC7 == 9)
replace VMB8 = . if (VMB8 == 99)
replace VMC8 = . if (VMC8 == 9)
replace VMB9 = . if (VMB9 == 99)
replace VMC9 = . if (VMC9 == 9)
replace VMB10 = . if (VMB10 == 99)
replace VMC10 = . if (VMC10 == 9)
replace VMB11 = . if (VMB11 == 99)
replace VMC11 = . if (VMC11 == 9)
replace VMB12 = . if (VMB12 == 99)
replace VMC12 = . if (VMC12 == 9)
replace VMB13 = . if (VMB13 == 99)
replace VMC13 = . if (VMC13 == 9)
replace VMB14 = . if (VMB14 == 99)
replace VMC14 = . if (VMC14 == 9)
replace VMB15 = . if (VMB15 == 99)
replace VMC15 = . if (VMC15 == 9)
replace VMB16 = . if (VMB16 == 99)
replace VMC16 = . if (VMC16 == 9)
replace VJ3A = . if (VJ3A == 9)
replace VJ3B = . if (VJ3B == 9)
replace VJ3C = . if (VJ3C == 9)
replace VJ3D = . if (VJ3D == 9)
replace VJ3E = . if (VJ3E == 9)
replace VJ3F = . if (VJ3F == 9)
replace VJ3G = . if (VJ3G == 9)
replace VJ3H = . if (VJ3H == 9)
replace VJ3I = . if (VJ3I == 9)


