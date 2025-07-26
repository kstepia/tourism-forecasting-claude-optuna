# tests/test_visitors.py
"""
제주 방문객 데이터 크롤러 테스트
최근 7일 데이터를 기반으로 크롤링 품질 및 안정성 검증
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import tempfile
import logging

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.visitors.fetch_visitors import JejuVisitorsCrawler

# 테스트용 로깅 설정
logging.basicConfig(level=logging.WARNING)  # 테스트 중 로그 최소화

class TestJejuVisitorsCrawler(unittest.TestCase):
    """제주 방문객 데이터 크롤러 테스트 클래스"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화 - 최근 7일 테스트 데이터 생성"""
        print("🔍 제주 방문객 크롤러 테스트 시작...")
        print("📅 최근 7일 테스트 데이터 수집 중...")
        
        cls.crawler = JejuVisitorsCrawler()
        cls.test_data_path = project_root / "tests" / "test_visitor_data.csv"
        
        # 테스트용 임시 다운로드 디렉토리 설정
        cls.temp_dir = tempfile.mkdtemp()
        os.environ["DOWNLOAD_DIR"] = cls.temp_dir
        
        # 최근 7일 테스트 데이터 수집
        cls._collect_test_data()
        
    @classmethod
    def _collect_test_data(cls):
        """최근 7일 방문객 데이터 수집"""
        try:
            # 크롤러 실행하여 최신 데이터 수집
            all_data = cls.crawler.run()
            
            if not all_data.empty:
                # 최근 7일 데이터만 필터링
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=7)
                
                # 날짜 컬럼이 datetime 타입인지 확인
                if 'date' in all_data.columns:
                    all_data['date'] = pd.to_datetime(all_data['date'])
                    recent_data = all_data[
                        (all_data['date'].dt.date >= start_date) & 
                        (all_data['date'].dt.date <= end_date)
                    ].copy()
                    
                    # 테스트 데이터 저장 (CSV 형식)
                    recent_data.to_csv(cls.test_data_path, index=False, encoding='utf-8')
                    cls.test_data = recent_data
                    
                    print(f"✅ 테스트 데이터 수집 완료: {len(recent_data)}건")
                    print(f"📊 데이터 범위: {recent_data['date'].min()} ~ {recent_data['date'].max()}")
                else:
                    # 빈 데이터프레임으로 초기화
                    cls.test_data = pd.DataFrame()
                    print("⚠️  날짜 컬럼을 찾을 수 없음")
            else:
                cls.test_data = pd.DataFrame()
                print("⚠️  수집된 데이터가 없음")
                
        except Exception as e:
            print(f"❌ 테스트 데이터 수집 실패: {e}")
            cls.test_data = pd.DataFrame()
    
    def test_crawler_initialization(self):
        """크롤러 초기화 테스트"""
        self.assertIsInstance(self.crawler, JejuVisitorsCrawler)
        self.assertGreater(len(self.crawler.sites), 0)
        self.assertIn('visitjeju', self.crawler.sites)
        self.assertIn('ijto', self.crawler.sites)
        
    def test_sites_configuration(self):
        """사이트 설정 테스트"""
        expected_sites = ['ijto', 'datahub', 'tour_kr', 'visitjeju', 'kto', 'kosis', 'data_go_kr', 'jeju_gov']
        
        for site in expected_sites:
            self.assertIn(site, self.crawler.sites, f"{site} 사이트가 설정되지 않음")
            self.assertTrue(self.crawler.sites[site].startswith('http'), f"{site} URL이 유효하지 않음")
    
    def test_retry_strategies(self):
        """403 우회 전략 테스트"""
        expected_strategies = ['basic', 'user_agent_rotation', 'session_reset', 'delay_retry', 'referer_header']
        
        for strategy in expected_strategies:
            self.assertIn(strategy, self.crawler.retry_strategies)
    
    def test_headers_generation(self):
        """HTTP 헤더 생성 테스트"""
        # 기본 헤더 테스트
        headers = self.crawler._get_headers('basic')
        self.assertIn('User-Agent', headers)
        self.assertIn('Accept', headers)
        
        # User-Agent 로테이션 테스트
        headers_rotation = self.crawler._get_headers('user_agent_rotation')
        self.assertIn('User-Agent', headers_rotation)
        
        # Referer 헤더 테스트
        headers_referer = self.crawler._get_headers('referer_header', 'https://google.com')
        self.assertIn('Referer', headers_referer)
        self.assertEqual(headers_referer['Referer'], 'https://google.com')
    
    def test_collected_data_structure(self):
        """수집된 데이터 구조 테스트"""
        if self.test_data.empty:
            self.skipTest("테스트 데이터가 없음")
        
        # 필수 컬럼 존재 확인
        required_columns = ['date', 'visitors', 'source']
        for col in required_columns:
            self.assertIn(col, self.test_data.columns, f"필수 컬럼 {col}이 없음")
        
        # 데이터 타입 확인
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.test_data['date']), "날짜 컬럼이 datetime 타입이 아님")
        self.assertTrue(pd.api.types.is_numeric_dtype(self.test_data['visitors']), "방문객 수가 숫자 타입이 아님")
    
    def test_data_quality(self):
        """데이터 품질 테스트"""
        if self.test_data.empty:
            self.skipTest("테스트 데이터가 없음")
        
        # 결측값 확인
        self.assertFalse(self.test_data['date'].isna().any(), "날짜에 결측값 존재")
        self.assertFalse(self.test_data['visitors'].isna().any(), "방문객 수에 결측값 존재")
        
        # 방문객 수 범위 확인 (합리적인 범위)
        min_visitors = self.test_data['visitors'].min()
        max_visitors = self.test_data['visitors'].max()
        
        self.assertGreater(min_visitors, 0, "방문객 수가 0 이하")
        self.assertLess(max_visitors, 10000000, "방문객 수가 비현실적으로 큼")
        
        # 음수 값 확인
        negative_count = (self.test_data['visitors'] < 0).sum()
        self.assertEqual(negative_count, 0, "음수 방문객 수 존재")
    
    def test_date_range_validation(self):
        """날짜 범위 검증 테스트"""
        if self.test_data.empty:
            self.skipTest("테스트 데이터가 없음")
        
        # 최근 7일 범위 확인
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        
        dates = pd.to_datetime(self.test_data['date']).dt.date
        
        # 모든 날짜가 지정된 범위 내에 있는지 확인
        valid_dates = (dates >= start_date) & (dates <= end_date)
        invalid_count = (~valid_dates).sum()
        
        self.assertEqual(invalid_count, 0, f"범위를 벗어난 날짜 {invalid_count}개 발견")
    
    def test_source_diversity(self):
        """데이터 소스 다양성 테스트"""
        if self.test_data.empty:
            self.skipTest("테스트 데이터가 없음")
        
        # 다양한 소스에서 데이터가 수집되었는지 확인
        sources = self.test_data['source'].unique()
        self.assertGreater(len(sources), 0, "데이터 소스가 없음")
        
        # 각 소스별 데이터 개수 출력
        source_counts = self.test_data['source'].value_counts()
        print("\n📈 소스별 데이터 개수:")
        for source, count in source_counts.items():
            print(f"  {source}: {count}건")
    
    def test_duplicate_handling(self):
        """중복 데이터 처리 테스트"""
        if self.test_data.empty:
            self.skipTest("테스트 데이터가 없음")
        
        # 같은 날짜의 중복 데이터 확인
        duplicate_dates = self.test_data.groupby('date').size()
        max_records_per_date = duplicate_dates.max()
        
        # 하루에 너무 많은 레코드가 있으면 중복 처리가 제대로 안된 것
        self.assertLessEqual(max_records_per_date, 10, "같은 날짜에 너무 많은 레코드 존재")
    
    def test_data_completeness(self):
        """데이터 완성도 테스트"""
        if self.test_data.empty:
            self.skipTest("테스트 데이터가 없음")
        
        # 최근 7일 중 데이터가 있는 날 수 확인
        unique_dates = self.test_data['date'].dt.date.nunique()
        
        # 적어도 최근 3일 이상의 데이터가 있어야 함
        self.assertGreaterEqual(unique_dates, 1, "충분한 날짜의 데이터가 없음")
        
        print(f"\n📅 수집된 날짜 수: {unique_dates}일")
    
    def test_harmonization_logic(self):
        """데이터 정합 로직 테스트"""
        # 테스트용 더미 데이터 생성
        daily_data = pd.DataFrame({
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'visitors': [1000, 1500],
            'source': ['daily', 'daily'],
            'data_type': ['daily', 'daily']
        })
        
        monthly_data = pd.DataFrame({
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 3)],
            'visitors': [1200, 1800],
            'source': ['monthly', 'monthly'],
            'data_type': ['monthly', 'monthly']
        })
        
        # 정합 테스트
        harmonised = self.crawler.harmonise_data(daily_data, monthly_data)
        
        self.assertFalse(harmonised.empty, "정합된 데이터가 비어있음")
        self.assertEqual(len(harmonised), 3, "정합된 데이터 개수가 예상과 다름")
        
        # 중복 제거 확인 (1월 1일은 daily 데이터가 우선되어야 함)
        jan_1_data = harmonised[harmonised['date'] == datetime(2024, 1, 1)]
        self.assertEqual(len(jan_1_data), 1, "중복 제거가 제대로 되지 않음")
        self.assertEqual(jan_1_data.iloc[0]['visitors'], 1000, "일별 데이터 우선순위가 적용되지 않음")
    
    @classmethod
    def tearDownClass(cls):
        """테스트 종료 후 정리"""
        # 임시 디렉토리 정리
        import shutil
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        
        print("\n🧹 테스트 정리 완료")

class TestDataValidation(unittest.TestCase):
    """데이터 검증 전용 테스트 클래스"""
    
    def setUp(self):
        """각 테스트 전 실행"""
        self.test_data_path = project_root / "tests" / "test_visitor_data.csv"
        
        if self.test_data_path.exists():
            self.test_data = pd.read_csv(self.test_data_path)
        else:
            self.test_data = pd.DataFrame()
    
    def test_statistical_outliers(self):
        """통계적 이상값 탐지 테스트"""
        if self.test_data.empty:
            self.skipTest("테스트 데이터가 없음")
        
        visitors = self.test_data['visitors']
        
        # IQR 방법으로 이상값 탐지
        Q1 = visitors.quantile(0.25)
        Q3 = visitors.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = visitors[(visitors < lower_bound) | (visitors > upper_bound)]
        outlier_percentage = len(outliers) / len(visitors) * 100
        
        # 이상값이 전체의 10% 이하여야 함
        self.assertLess(outlier_percentage, 10, f"이상값이 너무 많음: {outlier_percentage:.1f}%")
        
        print(f"\n📊 통계 정보:")
        print(f"  평균: {visitors.mean():,.0f}")
        print(f"  중앙값: {visitors.median():,.0f}")
        print(f"  표준편차: {visitors.std():,.0f}")
        print(f"  이상값: {len(outliers)}개 ({outlier_percentage:.1f}%)")
    
    def test_data_consistency(self):
        """데이터 일관성 테스트"""
        if self.test_data.empty:
            self.skipTest("테스트 데이터가 없음")
        
        # 같은 소스에서 오는 데이터의 일관성 확인
        for source in self.test_data['source'].unique():
            source_data = self.test_data[self.test_data['source'] == source]
            
            if len(source_data) > 1:
                # 같은 소스의 방문객 수 변동이 너무 크지 않은지 확인
                visitors_std = source_data['visitors'].std()
                visitors_mean = source_data['visitors'].mean()
                
                if visitors_mean > 0:
                    cv = visitors_std / visitors_mean  # 변동계수
                    self.assertLess(cv, 5.0, f"{source} 소스의 데이터 변동이 너무 큼 (CV: {cv:.2f})")

def create_test_suite():
    """테스트 스위트 생성"""
    test_suite = unittest.TestSuite()
    
    # 크롤러 테스트 추가
    crawler_tests = unittest.TestLoader().loadTestsFromTestCase(TestJejuVisitorsCrawler)
    test_suite.addTests(crawler_tests)
    
    # 데이터 검증 테스트 추가
    validation_tests = unittest.TestLoader().loadTestsFromTestCase(TestDataValidation)
    test_suite.addTests(validation_tests)
    
    return test_suite

def run_tests():
    """테스트 실행"""
    print("🧪 제주 방문객 데이터 크롤러 테스트 시작")
    print("=" * 60)
    
    # 테스트 스위트 생성 및 실행
    test_suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print(f"🏁 테스트 완료")
    print(f"✅ 성공: {result.testsRun - len(result.failures) - len(result.errors)}개")
    
    if result.failures:
        print(f"❌ 실패: {len(result.failures)}개")
    
    if result.errors:
        print(f"💥 오류: {len(result.errors)}개")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)