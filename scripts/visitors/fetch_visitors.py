# scripts/visitors/fetch_visitors.py
"""
1. 제주관광협회(visitjeju.or.kr)에서 ▸ 일일 위젯 ▸ 월별 게시판 XLS 를 크롤링한다.
2. 일별·월별 시계열을 harmonise() 로 정합 후 visitors_daily.parquet 로 저장한다.
3. Cloudflare 429 대비 cloudscraper, 1초 sleep 적용.
"""

import os
import time
import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import logging
import io
import re
import random
import requests

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 다운로드 디렉토리 설정
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "data/raw")
Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)

class JejuVisitorsCrawler:
    def __init__(self):
        # 403 우회를 위한 다양한 헤더 설정
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
        
        # User-Agent 로테이션
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        
        # 확장된 제주 관광 데이터 사이트들
        self.sites = {
            'ijto': 'https://data.ijto.or.kr',  # 제주관광빅데이터
            'datahub': 'https://www.jejudatahub.net',  # 제주데이터허브  
            'tour_kr': 'https://datalab.visitkorea.or.kr',  # 한국관광데이터랩
            'visitjeju': 'https://www.visitjeju.or.kr',  # 제주관광협회 (403 대응)
            'kto': 'https://kto.visitkorea.or.kr',  # 한국관광공사
            'kosis': 'https://kosis.kr',  # 국가통계포털
            'data_go_kr': 'https://www.data.go.kr',  # 공공데이터포털
            'jeju_gov': 'https://www.jeju.go.kr'  # 제주특별자치도
        }
        
        # 공통 CSS 셀렉터 패턴
        self.selectors = {
            'data_links': [
                'a[href*=".xls"]', 'a[href*=".xlsx"]', 'a[href*=".csv"]',
                '.file-download a', '.data-download a', '.download-link',
                'a:contains("다운로드")', 'a:contains("통계")', 'a:contains("방문객")'
            ],
            'visitor_numbers': [
                '.visitor-count', '.tourism-stat', '.stat-number', 
                '.visitor-number', '.count-display', '.data-value',
                '[class*="visitor"]', '[class*="count"]', '[class*="stat"]'
            ],
            'date_elements': [
                '.date', '.stat-date', '.data-date', '.period',
                '[class*="date"]', 'time', '.timestamp'
            ]
        }
        
        # 403 우회 전략별 재시도 설정
        self.retry_strategies = [
            'basic',
            'user_agent_rotation', 
            'session_reset',
            'delay_retry',
            'referer_header'
        ]
    
    def _get_headers(self, strategy='basic', referer=None):
        """전략에 따른 HTTP 헤더 생성"""
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        if strategy == 'user_agent_rotation':
            headers['User-Agent'] = random.choice(self.user_agents)
        elif strategy == 'referer_header' and referer:
            headers['Referer'] = referer
            headers['User-Agent'] = self.user_agents[0]
        else:
            headers['User-Agent'] = self.user_agents[0]
            
        return headers
    
    def _robust_request(self, url, max_retries=3):
        """403 오류 대응 견고한 요청"""
        last_error = None
        
        for strategy in self.retry_strategies:
            for attempt in range(max_retries):
                try:
                    headers = self._get_headers(strategy, referer='https://www.google.com')
                    
                    # 전략별 추가 설정
                    if strategy == 'session_reset':
                        # 새로운 세션 생성
                        self.scraper.close()
                        self.scraper = cloudscraper.create_scraper(
                            browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
                        )
                    elif strategy == 'delay_retry':
                        time.sleep(2 + attempt)  # 점진적 지연
                    
                    # 요청 실행
                    response = self.scraper.get(url, headers=headers, timeout=15)
                    
                    # 성공 조건 확인
                    if response.status_code == 200:
                        logger.debug(f"성공: {strategy} 전략으로 {url} 접근")
                        return response
                    elif response.status_code == 403:
                        logger.debug(f"403 오류: {strategy} 전략 실패 (시도 {attempt+1})")
                        time.sleep(1)
                        continue
                    else:
                        response.raise_for_status()
                        
                except Exception as e:
                    last_error = e
                    logger.debug(f"{strategy} 전략 오류 (시도 {attempt+1}): {e}")
                    time.sleep(1)
                    continue
        
        # 모든 전략 실패
        logger.warning(f"모든 접근 전략 실패: {url} - {last_error}")
        raise last_error if last_error else Exception("All retry strategies failed")
        
    def fetch_daily_widget_data(self):
        """다중 사이트에서 일일 데이터 크롤링"""
        logger.info("일일 방문객 데이터 크롤링 시작...")
        all_daily_data = []
        
        for site_name, base_url in self.sites.items():
            try:
                logger.info(f"{site_name} 사이트 크롤링...")
                
                # 견고한 요청으로 메인 페이지 접근
                response = self._robust_request(base_url)
                time.sleep(1)
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 방문객 수 데이터 찾기 (다양한 셀렉터 시도)
                daily_data = self._extract_visitor_data(soup, site_name)
                all_daily_data.extend(daily_data)
                
            except Exception as e:
                logger.warning(f"{site_name} 사이트 크롤링 실패: {e}")
                continue
        
        logger.info(f"총 일일 데이터 {len(all_daily_data)}건 수집 완료")
        return pd.DataFrame(all_daily_data)
    
    def _extract_visitor_data(self, soup, source):
        """HTML에서 방문객 데이터 추출"""
        daily_data = []
        
        # 다양한 패턴으로 방문객 수 찾기
        for visitor_selector in self.selectors['visitor_numbers']:
            try:
                elements = soup.select(visitor_selector)
                for elem in elements:
                    text = elem.get_text(strip=True)
                    
                    # 숫자만 추출 (콤마 제거)
                    import re
                    numbers = re.findall(r'[\d,]+', text)
                    
                    for num_str in numbers:
                        try:
                            visitors_count = int(num_str.replace(',', ''))
                            if visitors_count > 1000:  # 합리적인 방문객 수 필터
                                
                                # 날짜 찾기 (주변 요소에서)
                                date_elem = self._find_nearby_date(elem)
                                date_obj = self._parse_date(date_elem)
                                
                                if date_obj:
                                    daily_data.append({
                                        'date': date_obj,
                                        'visitors': visitors_count,
                                        'source': source
                                    })
                        except (ValueError, TypeError):
                            continue
                            
            except Exception as e:
                logger.debug(f"셀렉터 {visitor_selector} 처리 중 오류: {e}")
                continue
        
        return daily_data
    
    def _find_nearby_date(self, element):
        """요소 주변에서 날짜 찾기"""
        # 부모, 형제, 자식 요소에서 날짜 찾기
        search_elements = [element, element.parent] if element.parent else [element]
        
        # 형제 요소들 추가
        if element.parent:
            search_elements.extend(element.parent.find_all())
        
        for elem in search_elements[:10]:  # 최대 10개 요소만 확인
            if elem:
                text = elem.get_text(strip=True)
                if self._contains_date_pattern(text):
                    return text
        
        return None
    
    def _contains_date_pattern(self, text):
        """텍스트에 날짜 패턴이 있는지 확인"""
        import re
        date_patterns = [
            r'\d{4}[.-]\d{1,2}[.-]\d{1,2}',  # 2024-01-01
            r'\d{1,2}[.-]\d{1,2}[.-]\d{4}',  # 01-01-2024
            r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일',  # 2024년 1월 1일
            r'\d{1,2}월\s*\d{1,2}일',  # 1월 1일
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _parse_date(self, date_text):
        """다양한 형식의 날짜 파싱"""
        if not date_text:
            return None
        
        import re
        
        # 다양한 날짜 형식 파싱 시도
        date_formats = [
            r'(\d{4})[.-](\d{1,2})[.-](\d{1,2})',
            r'(\d{1,2})[.-](\d{1,2})[.-](\d{4})',
            r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
        ]
        
        for i, pattern in enumerate(date_formats):
            match = re.search(pattern, date_text)
            if match:
                try:
                    if i == 0:  # YYYY-MM-DD
                        year, month, day = match.groups()
                    elif i == 1:  # DD-MM-YYYY
                        day, month, year = match.groups()
                    else:  # Korean format
                        year, month, day = match.groups()
                    
                    return datetime(int(year), int(month), int(day))
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def fetch_monthly_board_xls(self):
        """다중 사이트에서 XLS/CSV 파일 크롤링"""
        logger.info("월별 통계 파일 크롤링 시작...")
        all_monthly_data = []
        
        for site_name, base_url in self.sites.items():
            try:
                logger.info(f"{site_name} 사이트 파일 검색...")
                
                # 사이트별 데이터 페이지 탐색
                data_urls = self._find_data_pages(base_url)
                
                for data_url in data_urls:
                    monthly_data = self._crawl_data_files(data_url, site_name)
                    all_monthly_data.extend(monthly_data)
                    
            except Exception as e:
                logger.warning(f"{site_name} 파일 크롤링 실패: {e}")
                continue
        
        if all_monthly_data:
            combined_monthly = pd.concat(all_monthly_data, ignore_index=True)
            logger.info(f"총 월별 데이터 {len(combined_monthly)}건 수집 완료")
            return combined_monthly
        else:
            return pd.DataFrame()
    
    def _find_data_pages(self, base_url):
        """데이터 다운로드 페이지 찾기"""
        data_pages = [base_url]  # 메인 페이지부터 시작
        
        try:
            response = self._robust_request(base_url)
            time.sleep(1)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 데이터, 통계, 다운로드 관련 링크 찾기
            data_keywords = ['데이터', '통계', '자료', '다운로드', 'data', 'statistics']
            
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # 키워드가 포함된 링크 찾기
                if any(keyword in text.lower() or keyword in href.lower() for keyword in data_keywords):
                    full_url = href if href.startswith('http') else f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                    if full_url not in data_pages:
                        data_pages.append(full_url)
                        
        except Exception as e:
            logger.debug(f"데이터 페이지 검색 실패 {base_url}: {e}")
        
        return data_pages[:5]  # 최대 5개 페이지만 확인
    
    def _crawl_data_files(self, url, source):
        """특정 페이지에서 데이터 파일 크롤링"""
        monthly_data = []
        
        try:
            response = self._robust_request(url)
            time.sleep(1)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 파일 링크 찾기
            file_links = self._find_file_links(soup, url)
            
            for file_url, file_type in file_links[:3]:  # 최대 3개 파일 처리
                logger.info(f"{file_type} 파일 처리: {file_url}")
                
                try:
                    if file_type in ['xls', 'xlsx']:
                        df = self._process_excel_file(file_url, source)
                    elif file_type == 'csv':
                        df = self._process_csv_file(file_url, source)
                    else:
                        continue
                    
                    if not df.empty:
                        monthly_data.append(df)
                        
                except Exception as e:
                    logger.warning(f"파일 처리 실패 {file_url}: {e}")
                    continue
                    
        except Exception as e:
            logger.debug(f"페이지 크롤링 실패 {url}: {e}")
        
        return monthly_data
    
    def _find_file_links(self, soup, base_url):
        """페이지에서 다운로드 가능한 파일 링크 찾기"""
        file_links = []
        
        # 다양한 방법으로 파일 링크 찾기
        selectors = [
            'a[href$=".xls"]', 'a[href$=".xlsx"]', 'a[href$=".csv"]',
            'a[href*=".xls"]', 'a[href*=".xlsx"]', 'a[href*=".csv"]'
        ]
        
        for selector in selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href', '')
                    text = link.get_text(strip=True)
                    
                    # 관광, 방문객 관련 파일만 필터링
                    if any(keyword in text for keyword in ['관광', '방문', '통계', '데이터', '여행']):
                        full_url = href if href.startswith('http') else f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                        
                        # 파일 확장자 확인
                        if href.endswith(('.xls', '.xlsx')):
                            file_type = 'xlsx'
                        elif href.endswith('.csv'):
                            file_type = 'csv'
                        else:
                            continue
                            
                        file_links.append((full_url, file_type))
                        
            except Exception as e:
                logger.debug(f"셀렉터 {selector} 처리 중 오류: {e}")
                continue
        
        return list(set(file_links))  # 중복 제거
    
    def _process_excel_file(self, file_url, source):
        """Excel 파일 처리"""
        try:
            file_response = self._robust_request(file_url)
            time.sleep(1)
            
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                tmp_file.write(file_response.content)
                tmp_path = tmp_file.name
            
            # Excel 파일 읽기 (여러 시트 시도)
            try:
                df = pd.read_excel(tmp_path, sheet_name=0)
            except:
                # 첫 번째 시트 실패시 다른 시트 시도
                with pd.ExcelFile(tmp_path) as xls:
                    for sheet_name in xls.sheet_names:
                        try:
                            df = pd.read_excel(tmp_path, sheet_name=sheet_name)
                            break
                        except:
                            continue
                    else:
                        return pd.DataFrame()
            
            # 임시 파일 삭제
            os.unlink(tmp_path)
            
            return self._clean_dataframe(df, source)
            
        except Exception as e:
            logger.warning(f"Excel 파일 처리 실패: {e}")
            return pd.DataFrame()
    
    def _process_csv_file(self, file_url, source):
        """CSV 파일 처리"""
        try:
            file_response = self._robust_request(file_url)
            time.sleep(1)
            
            # CSV 읽기 (여러 인코딩 시도)
            encodings = ['utf-8', 'euc-kr', 'cp949', 'utf-8-sig']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(io.StringIO(file_response.text), encoding=encoding)
                    break
                except:
                    continue
            else:
                return pd.DataFrame()
            
            return self._clean_dataframe(df, source)
            
        except Exception as e:
            logger.warning(f"CSV 파일 처리 실패: {e}")
            return pd.DataFrame()
    
    def _clean_dataframe(self, df, source):
        """데이터프레임 정제"""
        try:
            # 날짜와 방문객 수 컬럼 찾기
            date_columns = []
            visitor_columns = []
            
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['날짜', 'date', '일자', '기간']):
                    date_columns.append(col)
                elif any(keyword in col_lower for keyword in ['방문', '관광', 'visitor', '인원', '수']):
                    visitor_columns.append(col)
            
            if not date_columns or not visitor_columns:
                return pd.DataFrame()
            
            # 첫 번째 날짜, 방문객 컬럼 사용
            date_col = date_columns[0]
            visitor_col = visitor_columns[0]
            
            # 데이터 정제
            df_clean = df[[date_col, visitor_col]].dropna()
            df_clean.columns = ['date', 'visitors']
            
            # 날짜 변환
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
            df_clean = df_clean.dropna(subset=['date'])
            
            # 방문객 수 숫자 변환
            df_clean['visitors'] = pd.to_numeric(df_clean['visitors'].astype(str).str.replace(',', ''), errors='coerce')
            df_clean = df_clean.dropna(subset=['visitors'])
            
            # 합리적인 범위 필터링
            df_clean = df_clean[(df_clean['visitors'] > 100) & (df_clean['visitors'] < 1000000)]
            
            df_clean['source'] = source
            
            return df_clean
            
        except Exception as e:
            logger.warning(f"데이터프레임 정제 실패: {e}")
            return pd.DataFrame()
    
    def harmonise_data(self, daily_df, monthly_df):
        """일별·월별 시계열 데이터 정합"""
        logger.info("데이터 정합 시작...")
        
        try:
            # 빈 데이터프레임 처리
            if daily_df.empty and monthly_df.empty:
                logger.warning("정합할 데이터가 없습니다")
                return pd.DataFrame()
            
            # 데이터 결합
            all_data = []
            
            if not daily_df.empty:
                daily_df['data_type'] = 'daily'
                all_data.append(daily_df)
            
            if not monthly_df.empty:
                monthly_df['data_type'] = 'monthly'
                all_data.append(monthly_df)
            
            if not all_data:
                return pd.DataFrame()
            
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # 날짜별 중복 제거 (일별 데이터 우선)
            combined_df = combined_df.sort_values(['date', 'data_type'], ascending=[True, True])
            harmonised_df = combined_df.drop_duplicates(subset=['date'], keep='first')
            
            # 날짜 순 정렬
            harmonised_df = harmonised_df.sort_values('date').reset_index(drop=True)
            
            # 최종 컬럼 정리
            final_df = harmonised_df[['date', 'visitors', 'source', 'data_type']].copy()
            
            logger.info(f"정합 완료: {len(final_df)}건")
            return final_df
            
        except Exception as e:
            logger.error(f"데이터 정합 실패: {e}")
            return pd.DataFrame()
    
    def save_to_parquet(self, df, filename="visitors_daily.parquet"):
        """데이터를 parquet 형식으로 저장"""
        if df.empty:
            logger.warning("저장할 데이터가 없습니다")
            return
        
        try:
            output_path = Path(DOWNLOAD_DIR) / filename
            df.to_parquet(output_path, index=False)
            logger.info(f"데이터 저장 완료: {output_path}")
            logger.info(f"저장된 데이터 범위: {df['date'].min()} ~ {df['date'].max()}")
            
        except Exception as e:
            logger.error(f"데이터 저장 실패: {e}")
    
    def run(self):
        """전체 크롤링 프로세스 실행"""
        logger.info("제주 방문객 데이터 크롤링 시작")
        
        # 1. 일일 위젯 데이터 수집
        daily_data = self.fetch_daily_widget_data()
        
        # 2. 월별 게시판 XLS 수집
        monthly_data = self.fetch_monthly_board_xls()
        
        # 3. 제주관광협회 특별 처리
        try:
            visitjeju_data = self.fetch_visitjeju_special()
            if not visitjeju_data.empty:
                logger.info(f"제주관광협회 추가 데이터 {len(visitjeju_data)}건 수집")
                if monthly_data.empty:
                    monthly_data = visitjeju_data
                else:
                    monthly_data = pd.concat([monthly_data, visitjeju_data], ignore_index=True)
        except Exception as e:
            logger.warning(f"제주관광협회 특별 처리 실패: {e}")
        
        # 4. 데이터 정합
        harmonised_data = self.harmonise_data(daily_data, monthly_data)
        
        # 5. Parquet 저장
        self.save_to_parquet(harmonised_data)
        
        logger.info("크롤링 완료")
        return harmonised_data
    
    def fetch_visitjeju_special(self):
        """제주관광협회 사이트 특별 처리 (403 우회)"""
        logger.info("제주관광협회 사이트 특별 크롤링...")
        
        visitjeju_urls = [
            'https://www.visitjeju.or.kr/kr/tourInfo/statistics',
            'https://www.visitjeju.or.kr/kr/board/list',
            'https://www.visitjeju.or.kr/kr/tourInfo',
            'https://www.visitjeju.or.kr'
        ]
        
        all_data = []
        
        for url in visitjeju_urls:
            try:
                logger.info(f"제주관광협회 URL 시도: {url}")
                
                # 특별 헤더로 요청
                special_headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'ko-KR,ko;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'Referer': 'https://www.google.com/search?q=제주관광협회',
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache'
                }
                
                # requests 직접 사용 (cloudscraper 우회)
                session = requests.Session()
                session.headers.update(special_headers)
                
                response = session.get(url, timeout=15, verify=True)
                
                if response.status_code == 200:
                    logger.info(f"제주관광협회 접근 성공: {url}")
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # 방문객 데이터 추출
                    data = self._extract_visitor_data(soup, 'visitjeju_special')
                    all_data.extend(data)
                    
                    # 추가 데이터 페이지 탐색
                    more_urls = self._find_data_pages(url)
                    for more_url in more_urls[:2]:  # 최대 2개 추가 페이지
                        try:
                            more_response = session.get(more_url, timeout=10)
                            if more_response.status_code == 200:
                                more_soup = BeautifulSoup(more_response.content, 'html.parser')
                                more_data = self._extract_visitor_data(more_soup, 'visitjeju_special')
                                all_data.extend(more_data)
                        except:
                            continue
                    
                    time.sleep(2)  # 성공시 longer delay
                    break  # 성공하면 다른 URL 시도하지 않음
                    
                else:
                    logger.warning(f"제주관광협회 {response.status_code} 오류: {url}")
                    
            except Exception as e:
                logger.warning(f"제주관광협회 URL {url} 실패: {e}")
                continue
        
        logger.info(f"제주관광협회에서 {len(all_data)}건 데이터 수집")
        return pd.DataFrame(all_data)

def main():
    """메인 실행 함수"""
    crawler = JejuVisitorsCrawler()
    result = crawler.run()
    
    if not result.empty:
        print(f"수집 완료: {len(result)}건")
        print(f"데이터 범위: {result['date'].min()} ~ {result['date'].max()}")
        print(f"저장 위치: {DOWNLOAD_DIR}/visitors_daily.parquet")
    else:
        print("데이터 수집 실패")

if __name__ == "__main__":
    main()