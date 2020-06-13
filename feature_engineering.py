from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pandas as pd
import numpy as np
from .config import nans, outliers_bound


class NATransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.nans_handling_dict = nans
        self.col_medians = {}
        self.col_maxs = {}
        
    
    def fit(self, df):
        
        self._validate_nans_dict(df)
        fill_med_cols = self.nans_handling_dict['fill_median']
        for col in fill_med_cols:
            self.col_medians[col] = df[col].median()
        
        fill_max_cols = self.nans_handling_dict['fill_max']
        for col in fill_max_cols:
            self.col_maxs[col] = df[col].max()
                    
        return self
    
    def transform(self, df):
        
        df = df.copy()
        
        for na_method, cols in self.nans_handling_dict.items():
    
            if na_method == 'drop':
                df = df.dropna(subset=cols)

            elif na_method == 'fill_0':
                df.loc[:, cols] = df.loc[:, cols].fillna(0)

            elif na_method == 'fill_median':
                for col in cols:
                    med = self.col_medians[col]
                    df.loc[:, col] = df.loc[:, col].fillna(med)

            elif na_method == 'fill_max':
                for col in cols:
                    col_max = self.col_maxs[col]
                    df.loc[:, col] = df.loc[:, col].fillna(col_max)

            elif na_method == 'fill_dummy':
                for col in cols:
                    df.loc[:, col] = np.where(df[col].isnull(), 1, 0)

            else:
                print(f'{na_method}, not processed na col {cols}')
                    
        return df
    
    
    def _validate_nans_dict(self, df):
        df = df.copy()
        cols = df.columns.tolist()
        for na_method, na_cols in self.nans_handling_dict.items():
            unmatched_cols = list(set(na_cols) - set(cols))
            if len(unmatched_cols) > 0:
                matched_cols = list(set(na_cols).intersection(set(cols)))
                self.nans_handling_dict[na_method] = matched_cols
                print(f'{unmatched_cols} does not appear in {na_method} columns')
        
        return self

    
    
class OutlierTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.outlier_boundary_dict = outliers_bound
    
    
    def fit(self, df):
        
        return self
    
    def transform(self, df):
        
        for col in df.columns:
            try:
                col_outlier = self.outlier_boundary_dict[col]
                lower_bound = col_outlier['lower']
                upper_bound = col_outlier['upper']
                size_before = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                #print(f'col: {col} --> boundaries: [{lower_bound}, {upper_bound}], {size_before - len(df)} rows dropped')

            except KeyError:
                pass
    
        return df
    


class CatVarTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, drop=True, dummy=True, to_be_ignored=None):
        
        self.cat_cols = ['home_ownership', 'verification_status', 
                         'purpose',  'application_type', 'grade']
        if to_be_ignored is None:
            self.to_be_ignored = ['emp_title', 'sub_grade', 'title', 'zip_code', 'addr_state']
        else:
            self.to_be_ignored = to_be_ignored
            
        self.drop = drop
        self.unique_cat = {}
        self.dummy = dummy
        self.ohe = OneHotEncoder(drop='first', handle_unknown='error')
        
    
    def fit(self, df):
        
        if self.dummy:
            self.ohe.fit(df[self.cat_cols])

            for col in self.cat_cols:
                self.unique_cat[col] = df[col].unique().tolist()
        
        return self
    
    def transform(self, df):
        
        if self.dummy:
            # remove record containing new category
            for col in self.cat_cols:

                new_cat_df = df[~ df[col].isin(self.unique_cat[col])]
                new_cat = new_cat_df[col].unique().tolist()

                df = df[df[col].isin(self.unique_cat[col])]

                if len(new_cat) > 0:
                    print(f'col new category {new_cat} appear in {col} col, {new_cat_df.shape[0]} records removed')

            # transform 
            bin_matrix = self.ohe.transform(df[self.cat_cols]).toarray()        
            df_dummy = pd.DataFrame(bin_matrix, 
                                    columns=self.ohe.get_feature_names(self.cat_cols), 
                                    index=df.index)

            # append df by new data
            df = pd.concat([df, df_dummy], axis=1)            
            if self.drop:
                df = df.drop(columns=self.cat_cols)
                
        df = df.drop(columns=self.to_be_ignored)

        return df
    

class ContVarTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, log_transform=False, polynomial_term=False):
        self.log_transform = log_transform
        self.polynomial_term = polynomial_term
    
    def fit(self, df):
        
        return self
    
    
    def transform(self, df):
        
        df = self._add_year_since_earliest_cred_line(df)
        df = self._add_pmt_to_income(df)
        df = self._add_income_to_cred_lim(df)
        
        if self.log_transform:
            df = self._log_transform(df)
        
        if self.polynomial_term:
            df = self._add_poly_term(df)
        
        return df
    
    
    def _add_pmt_to_income(self, df):
        """
        """
        
        df.loc[:, 'pti'] = df['installment']/(df['annual_inc']/12)
        df.loc[:, 'pti'] = np.minimum(df['pti'], 100)
        
        return df
    
    
    def _add_income_to_cred_lim(self, df):
        
        """This is to validate if the income declared by the borrower makes sense
        """
    
        df.loc[:, 'income_to_cred_lim'] = np.divide(df['annual_inc'],
                                                   df['tot_hi_cred_lim'])

        df.loc[:, 'income_to_cred_lim'] = np.minimum(df['income_to_cred_lim'], 100)

        return df

    def _add_year_since_earliest_cred_line(self, df):
        
        report_date = datetime(2015, 12, 1)
        df.loc[:, 'yr_since_earliest_cred'] = (report_date - df['earliest_cr_line']).dt.days/365
        df.loc[:, 'yr_since_earliest_cred'] = np.minimum(df['yr_since_earliest_cred'], 90)
        df = df.drop(columns=['earliest_cr_line'])
        

        return df
    
    def _log_transform(self, df):
        
        log_cols = ['annual_inc', 'revol_bal', 'total_bal_ex_mort', 'tot_cur_bal', 'avg_cur_bal',
                     'bc_open_to_buy', 'tot_hi_cred_lim', 'total_rev_hi_lim', 'total_bc_limit',
                     'total_il_high_credit_limit', 'yr_since_earliest_cred']
        
        for col in log_cols:
            
            if col in df.columns.tolist():
                print(f'log transform {col}')
                df.loc[:, col] = np.log(df[col] + 0.01)
            else:
                pass

        return df
    
    def _add_poly_term(self, df):
        
        df_cont = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
        for col in df_cont.columns:
            df_cont.loc[:, col + '_poly2'] = np.square(df[col])
            
        df_others = df.select_dtypes(exclude=['float64', 'float32', 'int64', 'int32'])
        
        df = df_cont.merge(df_others, how='left', left_index=True, right_index=True)
        
        return df
    

class MyStandardScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_var = 'is_defaulted'
        self.feature_list = None

    def fit(self, df):
        self.feature_list = [col for col in df.columns if col != self.target_var]
        self.scaler.fit(df[self.feature_list])
        
        return self
    
    def transform(self, df):
        
        target_col = df[self.target_var].tolist()
        df = pd.DataFrame(self.scaler.transform(df[self.feature_list]),
                          columns=self.feature_list,
                          index = df.index)
        df.loc[:, self.target_var] = target_col
        
        return df
